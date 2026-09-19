"""Tests for the stdlib-only, offline, read-only checkpoint preflight."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PREFLIGHT = REPOSITORY / "preflight.py"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def safe_tensor_bytes(
    tensors: dict[str, dict[str, object]] | None = None,
    payload: bytes = b"\0\0\0\0",
) -> bytes:
    if tensors is None:
        tensors = {
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        }
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + payload


def write_safe_tensor(path: Path, tensors: dict[str, dict[str, object]] | None = None, payload: bytes = b"\0\0\0\0") -> None:
    path.write_bytes(safe_tensor_bytes(tensors, payload))


def valid_config(policy_type: str = "diffusion") -> dict[str, object]:
    return {
        "type": policy_type,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [2]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [2]},
        },
    }


def valid_processor(state_file: str, nested: bool = False) -> dict[str, object]:
    step: dict[str, object] = {"registry_name": "normalizer_processor"}
    if nested:
        step["config"] = {"state_file": state_file}
    else:
        step["state_file"] = state_file
    return {"name": "processor", "steps": [step]}


def make_valid_checkpoint(root: Path, policy_type: str = "diffusion") -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    write_json(checkpoint / "config.json", valid_config(policy_type))
    write_safe_tensor(checkpoint / "model.safetensors")
    write_json(checkpoint / "policy_preprocessor.json", valid_processor("pre.safetensors"))
    write_safe_tensor(checkpoint / "pre.safetensors")
    write_json(checkpoint / "policy_postprocessor.json", valid_processor("post.safetensors"))
    write_safe_tensor(checkpoint / "post.safetensors")
    return checkpoint


class PreflightCliTests(unittest.TestCase):
    maxDiff = None

    def run_preflight(self, checkpoint: Path, *arguments: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        result = subprocess.run(
            [sys.executable, str(PREFLIGHT), str(checkpoint), *arguments],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.stderr, "", result.stderr)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - produces useful test error
            self.fail(f"stdout was not JSON: {exc}: {result.stdout!r}")
        self.assertNotIn(str(checkpoint.resolve()), result.stdout)
        return result, report

    def assert_code(self, report: dict[str, object], code: str) -> None:
        checks = report["checks"]
        self.assertTrue(any(check["code"] == code for check in checks), checks)

    def test_expected_pass_for_diffusion_and_vqbet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for policy_type in ("diffusion", "vqbet"):
                checkpoint = make_valid_checkpoint(root / policy_type, policy_type)
                result, report = self.run_preflight(checkpoint)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["scope"]["supported_policy_family"], policy_type)
                self.assert_code(report, "SAFETENSORS_HEADER_VALID")
                self.assertIn("not a runtime pass", " ".join(report["boundaries"]))

    def test_supplied_lightweight_bundles_fail_only_for_missing_learned_weights(self) -> None:
        for checkpoint in (REPOSITORY / "checkpoint", REPOSITORY / "vqbet" / "checkpoint"):
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report["status"], "FAIL")
            self.assert_code(report, "SAFETENSORS_MISSING")
            failed_codes = {check["code"] for check in report["checks"] if check["status"] == "FAIL"}
            self.assertEqual(failed_codes, {"SAFETENSORS_MISSING"})

    def test_missing_and_malformed_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            (checkpoint / "config.json").unlink()
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "JSON_MISSING")

            checkpoint = make_valid_checkpoint(root / "malformed")
            (checkpoint / "config.json").write_text("{", encoding="utf-8")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "JSON_MALFORMED")

    def test_structural_empty_config_and_missing_model_regression_fixture(self) -> None:
        """A packaging symptom fixture; it does not reproduce or assign a cause."""
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_valid_checkpoint(Path(temporary))
            (checkpoint / "config.json").write_bytes(b"")
            (checkpoint / "model.safetensors").unlink()
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report["status"], "FAIL")
            failed_codes = {check["code"] for check in report["checks"] if check["status"] == "FAIL"}
            self.assertIn("JSON_MALFORMED", failed_codes)
            self.assertIn("SAFETENSORS_MISSING", failed_codes)

    def test_missing_malformed_and_invalid_processor_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            (checkpoint / "policy_preprocessor.json").unlink()
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "JSON_MISSING")

            checkpoint = make_valid_checkpoint(root / "malformed")
            (checkpoint / "policy_preprocessor.json").write_text("not json", encoding="utf-8")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "JSON_MALFORMED")

            checkpoint = make_valid_checkpoint(root / "invalid")
            write_json(checkpoint / "policy_preprocessor.json", {"steps": {}})
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "PROCESSOR_STEPS_INVALID")

    def test_missing_nested_and_unsafe_processor_state_references(self) -> None:
        unsafe_references = ("../outside.safetensors", "/absolute.safetensors", "C:\\drive.safetensors", "nested/file.safetensors")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            write_json(checkpoint / "policy_preprocessor.json", valid_processor("missing.safetensors", nested=True))
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "PROCESSOR_STATE_MISSING")

            for index, reference in enumerate(unsafe_references):
                checkpoint = make_valid_checkpoint(root / f"unsafe-{index}")
                write_json(checkpoint / "policy_preprocessor.json", valid_processor(reference))
                result, report = self.run_preflight(checkpoint)
                self.assertEqual(result.returncode, 1)
                self.assert_code(report, "PROCESSOR_STATE_PATH_UNSAFE")

    def test_symlinks_are_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            external = root / "external.safetensors"
            write_safe_tensor(external)
            (checkpoint / "pre.safetensors").unlink()
            os.symlink(external, checkpoint / "pre.safetensors")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "PROCESSOR_STATE_SYMLINK")

            checkpoint = make_valid_checkpoint(root / "model-link")
            external_model = root / "external-model.safetensors"
            write_safe_tensor(external_model)
            (checkpoint / "model.safetensors").unlink()
            os.symlink(external_model, checkpoint / "model.safetensors")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "SAFETENSORS_PATH_SYMLINK")

    def test_truncated_oversized_and_invalid_safetensors_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "truncated": struct.pack("<Q", 32) + b"{}",
                "oversized": struct.pack("<Q", 4 * 1024 * 1024 + 1),
                "invalid-json": struct.pack("<Q", 1) + b"[",
            }
            expected = {
                "truncated": "SAFETENSORS_HEADER_TRUNCATED",
                "oversized": "SAFETENSORS_HEADER_TOO_LARGE",
                "invalid-json": "SAFETENSORS_HEADER_MALFORMED",
            }
            for name, bytes_ in cases.items():
                checkpoint = make_valid_checkpoint(root / name)
                (checkpoint / "model.safetensors").write_bytes(bytes_)
                result, report = self.run_preflight(checkpoint)
                self.assertEqual(result.returncode, 1)
                self.assert_code(report, expected[name])

    def test_invalid_tensor_schema_offsets_and_config_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            bad_schema = {"weight": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}
            write_safe_tensor(checkpoint / "model.safetensors", bad_schema)
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "SAFETENSORS_SCHEMA_INVALID")

            checkpoint = make_valid_checkpoint(root / "offsets")
            bad_offsets = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 5]}}
            write_safe_tensor(checkpoint / "model.safetensors", bad_offsets, b"\0\0\0\0")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "SAFETENSORS_OFFSET_INVALID")

            checkpoint = make_valid_checkpoint(root / "shape")
            config = valid_config()
            config["input_features"]["observation.state"]["shape"] = [0]
            write_json(checkpoint / "config.json", config)
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "FEATURE_SCHEMA_INVALID")

    def test_unknown_policy_is_scoped_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_valid_checkpoint(Path(temporary), "other-policy")
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report["status"], "UNSUPPORTED")
            self.assert_code(report, "CONFIG_TYPE_UNSUPPORTED")
            self.assertFalse(report["scope"]["type_specific_checks"])

    def test_model_legacy_normalizer_keys_warn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_valid_checkpoint(Path(temporary))
            tensors = {
                "normalize_inputs.mean": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }
            write_safe_tensor(checkpoint / "model.safetensors", tensors)
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(report["status"], "WARN")
            self.assert_code(report, "LEGACY_NORMALIZER_KEYS_PRESENT")

    def test_optional_reports_are_external_new_paths_and_stdout_remains_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root)
            json_path = root / "report.json"
            markdown_path = root / "report.md"
            result, report = self.run_preflight(checkpoint, "--json", str(json_path), "--markdown", str(markdown_path))
            self.assertEqual(result.returncode, 0)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["status"], "PASS")
            self.assertIn("not** a runtime pass", markdown_path.read_text(encoding="utf-8"))

            # Existing files refuse overwrite and reports in the checkpoint are forbidden.
            existing = root / "existing.json"
            existing.write_text("preserve", encoding="utf-8")
            result, report = self.run_preflight(checkpoint, "--json", str(existing))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
            self.assert_code(report, "OUTPUT_PATH_REJECTED")

            inside = checkpoint / "forbidden.json"
            result, report = self.run_preflight(checkpoint, "--json", str(inside))
            self.assertEqual(result.returncode, 1)
            self.assertFalse(inside.exists())
            self.assert_code(report, "OUTPUT_PATH_REJECTED")

    def test_deep_and_duplicate_key_json_fail_without_tracebacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_valid_checkpoint(root / "deep")
            (checkpoint / "config.json").write_text("[" * 2000 + "0" + "]" * 2000)
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(any(c["code"] in {"JSON_MALFORMED", "JSON_SCHEMA_INVALID"} for c in report["checks"]))
            checkpoint = make_valid_checkpoint(root / "duplicate")
            (checkpoint / "config.json").write_text('{"type":"other","type":"diffusion"}')
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assert_code(report, "JSON_MALFORMED")
            checkpoint = make_valid_checkpoint(root / "header-deep")
            raw = ("[" * 2000 + "0" + "]" * 2000).encode()
            (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(any(c["code"] in {"SAFETENSORS_HEADER_MALFORMED", "SAFETENSORS_SCHEMA_INVALID"} for c in report["checks"]))

    def test_preflight_does_not_modify_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_valid_checkpoint(Path(temporary))
            before = {
                path.relative_to(checkpoint).as_posix(): (path.stat().st_size, path.read_bytes())
                for path in checkpoint.iterdir()
            }
            result, report = self.run_preflight(checkpoint)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(report["status"], "PASS")
            after = {
                path.relative_to(checkpoint).as_posix(): (path.stat().st_size, path.read_bytes())
                for path in checkpoint.iterdir()
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
