"""Build the verified checkpoint from pinned public weights and bundled processors.

SPDX-License-Identifier: Apache-2.0
The input file is never modified; the output directory must not exist.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file

SOURCE_REPO = "lerobot/vqbet_pusht"
SOURCE_REVISION = "15c2d0af889c401c7e5db7a07499d3b498afc276"
SOURCE_SHA = "f60b22049b275c026159fd4fcc018721ffbec47119ecfab98dc84814eb4e0d30"
MODEL_SHA = "2b1f2e478621f91a00c36cfa0dc06bf8f5ab1cc7d1bc1d86f58e53fec33b866b"


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Already-downloaded original model.safetensors")
    parser.add_argument("--output", type=Path, default=Path("vqbet-pusht"))
    parser.add_argument("--download-dir", type=Path, default=Path(".vqbet-pusht-source"),
                        help="Directory for the original weights; verified downloads can be reused")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists. Choose a new directory; existing files are not overwritten.")
    bundled = Path(__file__).resolve().parent / "checkpoint"
    expected = {}
    for line in (bundled / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if Path(name).name != name:
            raise ValueError("Unexpected path in checksum manifest")
        expected[name] = checksum
    if expected.get("model.safetensors") != MODEL_SHA:
        raise ValueError("Unexpected target model checksum")
    for name, checksum in expected.items():
        if name != "model.safetensors" and sha256(bundled / name) != checksum:
            raise ValueError(f"Bundled file checksum mismatch: {name}")
    source = args.source
    if source is None:
        from huggingface_hub import hf_hub_download

        print("Downloading the original public weights (approximately 158 MB).", flush=True)
        source = Path(hf_hub_download(
            SOURCE_REPO, "model.safetensors", revision=SOURCE_REVISION,
            local_dir=args.download_dir, token=False,
        ))
    if sha256(source) != SOURCE_SHA:
        raise ValueError("Source weights do not match the pinned official checkpoint")
    state = load_file(source, device="cpu")
    prefixes = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")
    core = {name: tensor for name, tensor in state.items() if not name.startswith(prefixes)}
    if len(core) != 207 or len(state) - len(core) != 6:
        raise ValueError("Unexpected model/statistic tensor count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pusht-prepare-", dir=args.output.parent) as temporary:
        staged = Path(temporary)
        save_file(core, staged / "model.safetensors", metadata={"format": "pt"})
        if sha256(staged / "model.safetensors") != MODEL_SHA:
            raise ValueError("Reconstructed model does not match the verified target")
        for name, checksum in expected.items():
            if name == "model.safetensors":
                continue
            shutil.copyfile(bundled / name, staged / name)
            if sha256(staged / name) != checksum:
                raise ValueError(f"Copied metadata checksum mismatch: {name}")
        shutil.copyfile(bundled / "SHA256SUMS", staged / "SHA256SUMS")
        # Exclusive creation prevents overwriting an existing checkpoint. If an I/O
        # failure occurs during publication, leave the partial directory for inspection.
        args.output.mkdir()
        for path in staged.iterdir():
            path.rename(args.output / path.name)
    print(json.dumps({
        "status": "PASS", "checkpoint": str(args.output),
        "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA,
        "model_sha256": MODEL_SHA, "core_tensors": len(core),
        "statistic_tensors_removed": len(state) - len(core),
        "all_packaged_file_checksums_match": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
