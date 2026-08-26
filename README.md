# ProtLST

**Protein Latent-Stage Trajectory**

This source package contains the training, inference, and attribution code for
the residue-preserving ProtLST VAE composite model. It does not include
datasets, cached embeddings, model checkpoints, or Qwen weights.

## Architecture

```text
ESM-C residues              [B,L,1152]
  -> Gaussian VAE posterior [B,L,256]
  -> z0/z1/z2/z3 trajectory [B,L,4,256]
  -> interleaved Bridge     [B,L*4,512]
  -> H1 local proxy / H2 domain proxy / H3 Function alignment
```

`z1` uses a residual depthwise local convolution, `z2` uses a two-layer
Transformer for long-range residue context, and `z3` applies a functional
projection. No stage removes the residue axis. The Bridge projects each stage
to 512 dimensions, adds residue and stage embeddings, and applies a two-layer
PyTorch TransformerEncoder.

The multitask loss combines balanced H1/H2 BCE, multi-positive Function
InfoNCE, z3-to-ESM reconstruction, and residue-level KL regularization.
UniProt features are proxy labels, not experimental 3D structure labels.

## Layout

- `prot_lst/protein_vae.py`: residue-level Gaussian VAE.
- `prot_lst/protein_vae_contrastive.py`: explicit z0/z1/z2/z3 states.
- `prot_lst/scripts/vae_trajectory/train_trajectory_vae.py`: base VAE training.
- `build_attribution_manifest.py`: deterministic split/label manifest.
- `build_esm_shards.py`: BF16 ESM-C residue cache.
- `train_attribution_experiment.py`: six experimental arms.
- `evaluate_attribution_experiment.py`: stage and shuffle evaluation.
- `run_attribution_matrix.py`: multi-arm, multi-seed launcher.
- `summarize_attribution_experiment.py`: aggregate report.
- `infer_trajectory.py`: sequence/FASTA to `[L,4,256]` trajectory.

## Installation

Use Python 3.10+ and install the project in editable mode:

```bash
python -m pip install -e .
prot-lst-train --help
prot-lst-infer --help
```

Alternatively, install `requirements.txt` and run the scripts directly. The
EvolutionaryScale ESM package must expose `esm.models.esmc.ESMC`. Depending on
your CUDA setup, install the matching PyTorch build before other dependencies.

## Train The Base VAE

The input JSONL requires `accession`, `sequence`, and optional `go_terms`,
`ec_terms`, and `function_text` fields.

```bash
prot-lst-train \
  --data data/uniprot.jsonl \
  --split-file data/splits.tsv \
  --split train \
  --esm-model esmc_600m \
  --limit 50000 \
  --steps 3000 \
  --batch-size 8 \
  --out runs/trajectory_vae.pt
```

## Infer A Trajectory

```bash
prot-lst-infer \
  --sequence MKT... \
  --checkpoint runs/trajectory_vae.pt \
  --output runs/example_trajectory.pt
```

The output stores one CPU tensor per protein with shape `[L,4,256]`.

## Attribution Training

Run `build_attribution_manifest.py`, then `build_esm_shards.py`, and finally
`run_attribution_matrix.py`. The six supported arms are `esm`, `vae_z3`,
`z3_transformer`, `trajectory_frozen`, `joint_function`, and
`joint_multitask`. Validation selects checkpoints; test evaluation should only
be run after the configuration is locked.

Frozen Qwen3-Embedding vectors are Function targets only. They are not protein
inputs, and GO/EC target labels must not be injected into the encoder path.
