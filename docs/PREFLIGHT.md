# Local checkpoint structural preflight

`preflight.py` is a **stdlib-only Python 3.10+** command-line check for a local LeRobot saved-checkpoint folder. It is intended to catch packaging problems before installing torch or downloading learned weights. It works offline and emits one structured JSON document to standard output.

```sh
python3 preflight.py /path/to/saved-checkpoint
```

A zero exit status means the structural result is `PASS` or `WARN`; a nonzero exit status means `FAIL` or scoped `UNSUPPORTED`. The supplied lightweight `checkpoint/` and `vqbet/checkpoint/` bundles are expected to report `FAIL` because they intentionally omit `model.safetensors`; their processor metadata can still be inspected.

Optional reports can be created only at **new paths outside the checkpoint**. Existing paths, symlinks, and paths inside the inspected checkpoint are refused.

```sh
python3 preflight.py /path/to/saved-checkpoint \
  --json /tmp/preflight.json --markdown /tmp/preflight.md
```

## What it checks

The preflight reads JSON artifacts and a bounded portion of SafeTensors files. It checks the `config.json` policy type (within the `diffusion` and `vqbet` scope), non-empty input/output feature shapes, required `model.safetensors`, processor manifests, and processor `state_file` references. State references must be safe basenames; traversal, absolute/drive-qualified paths, symlinks, paths resolving outside the checkpoint, and missing files are reported. SafeTensors inspection reads the 8-byte length prefix and at most a **4 MiB** UTF-8 JSON header, validates descriptor shapes and nonnegative file-bounded offsets, and does not load tensor data. It warns when a model header has the legacy normalizer prefixes `normalize_inputs.`, `normalize_targets.`, or `unnormalize_outputs.`.

The JSON report includes installed-version metadata for selected packages using `importlib.metadata`; it does not import torch, LeRobot, safetensors, diffusers, or any checkpoint code.

## Boundaries

This is **not a runtime pass**. It performs no inference, executes no checkpoint code, imports no third-party packages, makes no network request, and does not modify the checkpoint. A structural pass does **not** establish numerical compatibility, policy behavior, task success, or robot safety. The tool does not repair files. It is not a complete SafeTensors validator: dtype/byte-size agreement, contiguous payload coverage and tensor values are not checked. Processor step registry semantics are not checked against LeRobot. Use it on a stable local directory; it is not a sandbox for an adversarial, concurrently modified filesystem. Python 3.12.3 was exercised here; the 3.10+ syntax target has not been tested across a version matrix.

The test suite includes a **structural fixture** for the symptom of an empty `config.json` together with a missing `model.safetensors`, and asserts that both diagnostics are reported in one run. That fixture is only a packaging-shape regression check; it does not reproduce a particular release, identify a root cause, or claim a fix.

## Run tests

```sh
python3 -m unittest discover -s tests -v
```
