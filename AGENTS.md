# Repository Guidelines

## Project Structure & Module Organization

- `prot_lst/` contains the installable Python package and model implementations.
- `prot_lst/protein_vae.py` defines the residue-level Gaussian VAE; `protein_vae_contrastive.py` defines the staged `z0`–`z3` model.
- `prot_lst/scripts/vae_trajectory/` contains command-line workflows for training, inference, caching, attribution experiments, evaluation, and reporting.
- `README.md` documents the data schema and end-to-end experiment commands. Datasets, checkpoints, cached embeddings, and model weights are intentionally kept outside the repository (typically under local `data/` and `runs/`).

## Build, Test, and Development Commands

Use Python 3.10 or newer and install the package in editable mode:

```bash
python -m pip install -e .
prot-lst-train --help
prot-lst-infer --help
```

Run a CPU smoke test after changes to the training path:

```bash
prot-lst-train --data data/uniprot.jsonl --limit 32 --steps 5 \
  --batch-size 2 --device cpu --out runs/smoke_vae.pt
```

There is currently no committed test suite or build script. Exercise changed CLI workflows with small, local inputs and document any GPU-dependent validation.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and clear type-oriented names for tensors and CLI arguments. Keep model stages and tensor shapes explicit in comments or docstrings. Preserve deterministic ordering in manifests and experiment outputs. No formatter or linter is configured; keep imports clean and run `python -m compileall prot_lst` before submitting.

## Testing Guidelines

For model or script changes, run `python -m compileall prot_lst` plus the smallest relevant CLI smoke command. Verify tensor shapes, output files, and CPU loading where applicable. If adding tests, place them under `tests/`, name files `test_*.py`, and use `pytest`; avoid requiring downloaded datasets or model weights in unit tests.

## Commit & Pull Request Guidelines

Existing commits use short, imperative summaries such as `Update ...` and `Add ...`. Keep commits focused and describe the behavior changed. Pull requests should include a concise rationale, affected commands or modules, reproduction/validation commands, data or checkpoint assumptions, and relevant metrics or output-shape checks. Include screenshots only when documenting a user-facing visualization; do not commit large datasets, caches, or weights.

## Data and Configuration

Do not commit UniProt exports, ESM/Qwen caches, credentials, or checkpoints. Use versioned split files for reproducible experiments, keep generated artifacts under `runs/`, and record model names, seeds, device, and key hyperparameters when reporting results.
