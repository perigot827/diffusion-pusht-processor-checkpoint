#!/usr/bin/env python3
"""Offline, read-only structural preflight for a local LeRobot saved checkpoint.

This tool reads JSON and bounded SafeTensors headers only. It never imports a
checkpoint, imports third-party packages, runs inference, accesses a network,
or writes in the checkpoint directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import struct
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

FORMAT_VERSION = 1
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 4 * 1024 * 1024
SUPPORTED_POLICY_TYPES = {"diffusion", "vqbet"}
PROCESSOR_MANIFESTS = (
    ("policy_preprocessor.json", "preprocessor manifest"),
    ("policy_postprocessor.json", "postprocessor manifest"),
)
PACKAGE_NAMES = ("lerobot", "torch", "safetensors", "diffusers")


class Reporter:
    """Collect diagnostics without retaining user-provided file contents."""

    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []
        self.scope_supported = True
        self.policy_family: str | None = None

    def add(self, code: str, severity: str, status: str, message: str, subject: str | None = None) -> None:
        check = {
            "code": code,
            "severity": severity,
            "status": status,
            "message": message,
        }
        if subject is not None:
            check["subject"] = subject
        self.checks.append(check)

    def passed(self, code: str, message: str, subject: str | None = None) -> None:
        self.add(code, "INFO", "PASS", message, subject)

    def warning(self, code: str, message: str, subject: str | None = None) -> None:
        self.add(code, "WARNING", "WARN", message, subject)

    def failed(self, code: str, message: str, subject: str | None = None) -> None:
        self.add(code, "ERROR", "FAIL", message, subject)

    def unsupported(self, code: str, message: str, subject: str | None = None) -> None:
        self.scope_supported = False
        self.add(code, "WARNING", "UNSUPPORTED", message, subject)


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-standard JSON constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_safe_basename(reference: Any) -> bool:
    """Accept exactly one portable filename component, never a path."""
    if not isinstance(reference, str) or not reference or "\x00" in reference:
        return False
    if reference in {".", ".."} or "/" in reference or "\\" in reference:
        return False
    windows = PureWindowsPath(reference)
    if windows.is_absolute() or windows.drive:
        return False
    return Path(reference).name == reference


def _read_json_object(path: Path, reporter: Reporter, subject: str) -> dict[str, Any] | None:
    """Read a small regular JSON object, reporting only generic diagnostics."""
    if path.is_symlink():
        reporter.failed("JSON_PATH_SYMLINK", "A required JSON artifact is a symlink and was not followed.", subject)
        return None
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        reporter.failed("JSON_MISSING", "A required JSON artifact is missing.", subject)
        return None
    except OSError:
        reporter.failed("JSON_UNREADABLE", "A required JSON artifact could not be inspected.", subject)
        return None
    if not path.is_file():
        reporter.failed("JSON_NOT_REGULAR_FILE", "A required JSON artifact is not a regular file.", subject)
        return None
    if size > MAX_JSON_BYTES:
        reporter.failed("JSON_TOO_LARGE", "A JSON artifact exceeds the bounded inspection limit.", subject)
        return None
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            reporter.failed("JSON_TOO_LARGE", "A JSON artifact exceeds the bounded inspection limit.", subject)
            return None
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        reporter.failed("JSON_MALFORMED", "A JSON artifact is malformed or is not UTF-8 JSON.", subject)
        return None
    if not isinstance(value, dict):
        reporter.failed("JSON_SCHEMA_INVALID", "A JSON artifact must contain an object at its top level.", subject)
        return None
    return value


def _validate_feature_collection(config: dict[str, Any], field: str, reporter: Reporter) -> None:
    subject = "input features" if field == "input_features" else "output features"
    features = config.get(field)
    valid = isinstance(features, dict) and bool(features)
    if valid:
        for name, feature in features.items():
            if not isinstance(name, str) or not name or not isinstance(feature, dict):
                valid = False
                break
            feature_type = feature.get("type")
            shape = feature.get("shape")
            if not isinstance(feature_type, str) or not feature_type:
                valid = False
                break
            if not isinstance(shape, list) or not shape or not all(_is_positive_int(dimension) for dimension in shape):
                valid = False
                break
    if valid:
        reporter.passed("FEATURE_SCHEMA_VALID", "Feature definitions have non-empty, positive integer shapes.", subject)
    else:
        reporter.failed("FEATURE_SCHEMA_INVALID", "Feature definitions require non-empty, positive integer shapes and typed entries.", subject)


def _validate_config(checkpoint: Path, reporter: Reporter) -> None:
    config = _read_json_object(checkpoint / "config.json", reporter, "config.json")
    if config is None:
        reporter.unsupported(
            "CONFIG_SCOPE_UNAVAILABLE",
            "The policy configuration could not be validated, so type-specific checks were skipped.",
            "config.json",
        )
        return

    policy_type = config.get("type")
    if not isinstance(policy_type, str) or policy_type not in SUPPORTED_POLICY_TYPES:
        reporter.unsupported(
            "CONFIG_TYPE_UNSUPPORTED",
            "The policy type is absent or outside this preflight's diffusion/VQ-BeT scope; type-specific checks were skipped.",
            "config.json",
        )
        return

    reporter.policy_family = policy_type
    reporter.passed("CONFIG_TYPE_SUPPORTED", "The checkpoint declares a policy family covered by this preflight.", "config.json")
    _validate_feature_collection(config, "input_features", reporter)
    _validate_feature_collection(config, "output_features", reporter)


def _validate_safetensors(path: Path, reporter: Reporter, subject: str, inspect_legacy_keys: bool = False) -> None:
    """Validate bounded SafeTensors header structure without reading tensor data."""
    if path.is_symlink():
        reporter.failed("SAFETENSORS_PATH_SYMLINK", "A SafeTensors artifact is a symlink and was not followed.", subject)
        return
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        reporter.failed("SAFETENSORS_MISSING", "A required SafeTensors artifact is missing.", subject)
        return
    except OSError:
        reporter.failed("SAFETENSORS_UNREADABLE", "A SafeTensors artifact could not be inspected.", subject)
        return
    if not path.is_file():
        reporter.failed("SAFETENSORS_NOT_REGULAR_FILE", "A SafeTensors artifact is not a regular file.", subject)
        return
    if file_size < 8:
        reporter.failed("SAFETENSORS_TOO_SHORT", "A SafeTensors artifact is shorter than its fixed header-length field.", subject)
        return

    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                reporter.failed("SAFETENSORS_TOO_SHORT", "A SafeTensors artifact is shorter than its fixed header-length field.", subject)
                return
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                reporter.failed("SAFETENSORS_HEADER_TOO_LARGE", "The declared SafeTensors header exceeds the bounded inspection limit.", subject)
                return
            if header_size > file_size - 8:
                reporter.failed("SAFETENSORS_HEADER_TRUNCATED", "The declared SafeTensors header is truncated.", subject)
                return
            header_bytes = stream.read(header_size)
    except OSError:
        reporter.failed("SAFETENSORS_UNREADABLE", "A SafeTensors artifact could not be inspected.", subject)
        return

    if len(header_bytes) != header_size:
        reporter.failed("SAFETENSORS_HEADER_TRUNCATED", "The declared SafeTensors header is truncated.", subject)
        return
    try:
        header = json.loads(header_bytes.decode("utf-8"), parse_constant=_reject_json_constant, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        reporter.failed("SAFETENSORS_HEADER_MALFORMED", "The SafeTensors header is not valid UTF-8 JSON.", subject)
        return
    if not isinstance(header, dict):
        reporter.failed("SAFETENSORS_SCHEMA_INVALID", "The SafeTensors header must contain an object.", subject)
        return

    payload_size = file_size - 8 - header_size
    ranges: list[tuple[int, int]] = []
    tensor_names: list[str] = []
    schema_valid = True
    offset_valid = True
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                schema_valid = False
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            schema_valid = False
            continue
        tensor_names.append(name)
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype or not isinstance(shape, list) or not all(
            _is_nonnegative_int(dimension) for dimension in shape
        ):
            schema_valid = False
            continue
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_nonnegative_int(value) for value in offsets):
            offset_valid = False
            continue
        start, end = offsets
        if start > end or end > payload_size:
            offset_valid = False
            continue
        ranges.append((start, end))

    if not tensor_names:
        schema_valid = False
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            offset_valid = False
            break

    if not schema_valid:
        reporter.failed("SAFETENSORS_SCHEMA_INVALID", "The SafeTensors header has an invalid tensor descriptor schema or shape.", subject)
        return
    if not offset_valid:
        reporter.failed("SAFETENSORS_OFFSET_INVALID", "The SafeTensors header has non-integer, overlapping, or out-of-bounds tensor offsets.", subject)
        return

    reporter.passed("SAFETENSORS_HEADER_VALID", "The bounded SafeTensors header passes basic descriptor and file-bounded offset checks; dtype sizes and payload contents are not validated.", subject)
    if inspect_legacy_keys:
        legacy_count = sum(
            name.startswith(("normalize_inputs.", "normalize_targets.", "unnormalize_outputs."))
            for name in tensor_names
        )
        if legacy_count:
            reporter.warning(
                "LEGACY_NORMALIZER_KEYS_PRESENT",
                "The model header contains legacy normalizer tensor-name prefixes; saved processor state should be reviewed.",
                subject,
            )


def _state_file_references(value: Any):
    """Walk decoded JSON iteratively; deep input never recurses here."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "state_file":
                    yield child
                pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)


def _validate_processors(checkpoint: Path, reporter: Reporter) -> None:
    root = checkpoint.resolve(strict=False)
    for filename, label in PROCESSOR_MANIFESTS:
        manifest = _read_json_object(checkpoint / filename, reporter, filename)
        if manifest is None:
            continue
        steps = manifest.get("steps")
        if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
            reporter.failed("PROCESSOR_STEPS_INVALID", "A processor manifest must contain a list of step objects.", filename)
            continue

        reporter.passed("PROCESSOR_MANIFEST_VALID", "The processor manifest is valid JSON with a step list.", filename)
        for reference in _state_file_references(manifest):
            if not _is_safe_basename(reference):
                reporter.failed(
                    "PROCESSOR_STATE_PATH_UNSAFE",
                    "A processor state-file reference is not a safe basename; paths, traversal, absolute references, and drive-qualified names are rejected.",
                    label,
                )
                continue
            state_path = checkpoint / reference
            if state_path.is_symlink():
                reporter.failed("PROCESSOR_STATE_SYMLINK", "A processor state file is a symlink and was not followed.", label)
                continue
            # The reference is a basename, but resolve again to defend against
            # filesystem links and unusual parent layouts without exposing paths.
            try:
                resolved_state = state_path.resolve(strict=False)
                resolved_state.relative_to(root)
            except (OSError, ValueError):
                reporter.failed("PROCESSOR_STATE_OUTSIDE_CHECKPOINT", "A processor state-file reference resolves outside the checkpoint.", label)
                continue
            if not os.path.lexists(state_path):
                reporter.failed("PROCESSOR_STATE_MISSING", "A processor manifest references a missing state file.", label)
                continue
            _validate_safetensors(state_path, reporter, "processor state file")


def _package_metadata() -> dict[str, dict[str, str | bool | None]]:
    packages: dict[str, dict[str, str | bool | None]] = {}
    for name in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
        except Exception:
            # Metadata discovery is non-essential and must not prevent structural checks.
            packages[name] = {"installed": False, "version": None}
        else:
            packages[name] = {"installed": True, "version": version}
    return packages


def _overall_status(reporter: Reporter) -> str:
    statuses = {check["status"] for check in reporter.checks}
    if "FAIL" in statuses:
        return "FAIL"
    if not reporter.scope_supported or "UNSUPPORTED" in statuses:
        return "UNSUPPORTED"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def preflight_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Return a report for a local folder using reads only; never raise on ordinary damage."""
    reporter = Reporter()
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink():
        reporter.warning("CHECKPOINT_ROOT_SYMLINK", "The checkpoint root is a symlink; contained artifacts are still checked without writing.")
    if not checkpoint.exists() or not checkpoint.is_dir():
        reporter.failed("CHECKPOINT_NOT_DIRECTORY", "The requested checkpoint is not an accessible directory.")
    else:
        _validate_config(checkpoint, reporter)
        _validate_safetensors(checkpoint / "model.safetensors", reporter, "model.safetensors", inspect_legacy_keys=True)
        _validate_processors(checkpoint, reporter)

    return {
        "format_version": FORMAT_VERSION,
        "status": _overall_status(reporter),
        "scope": {
            "supported_policy_family": reporter.policy_family,
            "type_specific_checks": reporter.scope_supported and reporter.policy_family is not None,
        },
        "checks": reporter.checks,
        "installed_packages": {
            "metadata_only": True,
            "packages": _package_metadata(),
        },
        "boundaries": [
            "This is a structural packaging preflight, not a runtime pass.",
            "It performs no inference, executes no checkpoint code, imports no third-party package, and makes no network request.",
            "It does not establish numerical compatibility, policy behavior, task success, or robot safety.",
            "SafeTensors inspection reads only the fixed prefix and at most a 4 MiB JSON header; tensor data are not loaded.",
            "This is not a complete SafeTensors validator: dtype/byte-size agreement, contiguous payload coverage and tensor values are not checked.",
        ],
    }


def _output_path_error(path: Path, checkpoint: Path) -> str | None:
    """Return a generic error for a forbidden output path, never the raw path."""
    try:
        candidate = path.expanduser().resolve(strict=False)
        root = checkpoint.expanduser().resolve(strict=False)
        candidate.relative_to(root)
    except ValueError:
        pass
    except OSError:
        return "An output path could not be resolved safely."
    else:
        return "An output path is inside the checkpoint and is refused."
    if os.path.lexists(path.expanduser()):
        return "An output path already exists or is a symlink and is refused."
    return None


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# LeRobot checkpoint structural preflight",
        "",
        f"**Overall status:** `{report['status']}`",
        "",
        "This report is a structural packaging preflight, **not** a runtime pass. It does not establish numerical compatibility, policy behavior, task success, or robot safety.",
        "",
        "| Code | Severity | Status | Subject | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        subject = check.get("subject", "")
        # Messages and subjects are produced internally, but escape defensively.
        row = [check["code"], check["severity"], check["status"], subject, check["message"]]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in report["boundaries"])
    lines.append("")
    return "\n".join(lines)


def _write_output(path: Path, content: str) -> bool:
    """Write only an explicitly requested, prevalidated external report path."""
    try:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # x mode keeps the no-overwrite promise even if a file appears after validation.
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        return True
    except (OSError, UnicodeError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline, read-only structural preflight for a local LeRobot saved checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Local saved-checkpoint directory to inspect")
    parser.add_argument("--json", dest="json_output", type=Path, help="Write the JSON report to a new path outside the checkpoint")
    parser.add_argument("--markdown", type=Path, help="Write a Markdown report to a new path outside the checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = [path for path in (args.json_output, args.markdown) if path is not None]
    reporter_errors: list[str] = []
    if len({str(path.expanduser().resolve(strict=False)) for path in output_paths}) != len(output_paths):
        reporter_errors.append("Requested output paths must be distinct.")
    for output_path in output_paths:
        error = _output_path_error(output_path, args.checkpoint)
        if error:
            reporter_errors.append(error)

    report = preflight_checkpoint(args.checkpoint)
    for error in dict.fromkeys(reporter_errors):
        report["checks"].append({
            "code": "OUTPUT_PATH_REJECTED",
            "severity": "ERROR",
            "status": "FAIL",
            "message": error,
        })
    if reporter_errors:
        report["status"] = "FAIL"

    # JSON stdout is emitted exactly once and contains no filesystem paths.
    json_stdout = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if not reporter_errors:
        failures: list[str] = []
        if args.json_output is not None and not _write_output(args.json_output, json_stdout):
            failures.append("The requested JSON output could not be written without overwriting an existing file.")
        if args.markdown is not None and not _write_output(args.markdown, _markdown_report(report)):
            failures.append("The requested Markdown output could not be written without overwriting an existing file.")
        if failures:
            for message in failures:
                report["checks"].append({
                    "code": "OUTPUT_WRITE_FAILED",
                    "severity": "ERROR",
                    "status": "FAIL",
                    "message": message,
                })
            report["status"] = "FAIL"
            json_stdout = json.dumps(report, indent=2, sort_keys=True) + "\n"

    sys.stdout.write(json_stdout)
    return 0 if report["status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
