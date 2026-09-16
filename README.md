# ProtLST

**Protein Latent-Stage Trajectory**

This source package contains the training, inference, and attribution code for
the residue-preserving ProtLST VAE composite model. It does not include
datasets, cached embeddings, model checkpoints, or Qwen weights.

## Architecture

```text
ESM-C 300M residues         [B,L,960]
  -> Gaussian VAE posterior [B,L,256]
  -> z0/z1/z2/z3 trajectory [B,L,4,256]
  -> stage-causal Bridge    [B,L*4,256]
  -> H1 local proxy / H2 domain proxy / H3 Function alignment
  -> z4 ESM reconstruction [B,L,960]
```

`z1` uses a residual depthwise local convolution, `z2` uses a two-layer
Transformer for long-range residue context, and `z3` applies a functional
projection. No stage removes the residue axis. The Bridge projects each stage
to 256 dimensions in the current trained configuration, adds residue and stage
embeddings, and applies a two-layer
PyTorch TransformerEncoder. Bridge attention is causal over stages but global
over residues: `Hk` may read all valid residues from `z0` through `zk`, never a
future stage. During training the trajectory starts from a reparameterized VAE
sample; inference and evaluation use the posterior mean.
The trajectory VAE has no amino-acid sequence decoder; its decoder is the
Bridge reconstruction head that maps H3 to the z4 ESM embedding estimate.

The multitask loss combines balanced H1/H2 BCE, paired positive-only Function
cosine alignment, H3-to-z4 ESM reconstruction, and residue-level KL
regularization. Function training uses no in-batch or queued negatives.
`z4` is a reconstruction output and is not an input stage of the Bridge.
UniProt features are proxy labels, not experimental 3D structure labels.
H1 is a fourteen-label residue-level sigmoid head (nine core labels, four
expanded biological labels, plus `other`) and H2 is a fifteen-label
domain/region sigmoid head (ten core labels, four expanded biological labels,
plus `other`).
Labels are independent, so overlapping annotations at one residue are preserved
rather than collapsed into one target. The two `other` labels pool non-core
UniProt point and interval annotations separately. `chain` is intentionally
excluded because it is near-universal entry-boundary metadata rather than a
discriminative biological feature. The canonical mapping lives in
`feature_taxonomy.py`; `analyze_feature_types.py` reproduces the full local-data
frequency report.

## Layout

- `prot_lst/protein_vae.py`: residue-level Gaussian VAE.
- `prot_lst/protein_vae_contrastive.py`: explicit z0/z1/z2/z3 states.
- `prot_lst/scripts/vae_trajectory/train_trajectory_vae.py`: base VAE training.
- `build_stage2_split.py`: duplicate-safe 45k/5k/5k split builder.
- `build_attribution_manifest.py`: header-driven split/label manifest.
- `build_esm_shards.py`: BF16 ESM-C residue cache.
- `train_attribution_experiment.py`: six experimental arms.
- `evaluate_attribution_experiment.py`: stage and shuffle evaluation.
- `run_attribution_matrix.py`: multi-arm, multi-seed launcher.
- `summarize_attribution_experiment.py`: aggregate report.
- `infer_trajectory.py`: sequence/FASTA to `[L,4,256]` trajectory.
- `cache_text_embeddings.py`: frozen Function-only text embedding cache.
- `run_stage2_joint_pipeline.py`: resumable 55k cache and four-epoch joint pipeline.

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

## Download And Prepare UniProt Data

The released code does not bundle protein data. The experiments used reviewed
Swiss-Prot records from the UniProt 2024_01 release. For a current reviewed
dataset, download the official UniProtKB JSON stream and save it locally:

```bash
mkdir -p data
curl --fail --location --retry 3 \
  --output data/uniprot_reviewed.json \
  'https://rest.uniprot.org/uniprotkb/stream?query=reviewed%3Atrue&format=json'
```

The REST response is a JSON object containing a `results` array, not the
training JSONL schema used by ProtLST. Convert it to JSONL with the following
small script (the sequence and accession are required; annotations are
optional):

```bash
python - <<'PY'
import json
from pathlib import Path

src = json.loads(Path("data/uniprot_reviewed.json").read_text())
with Path("data/uniprot.jsonl").open("w", encoding="utf-8") as out:
    for item in src.get("results", []):
        seq = item.get("sequence", {})
        desc = item.get("proteinDescription", {})
        comments = item.get("comments", [])
        function = " ".join(
            t.get("value", "")
            for c in comments if c.get("commentType") == "FUNCTION"
            for t in c.get("texts", []) if isinstance(t, dict)
        )
        go_terms = []
        for ref in item.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                go_terms.append(ref.get("id"))
        row = {
            "accession": item.get("primaryAccession"),
            "sequence": seq.get("value", ""),
            "length": seq.get("length", len(seq.get("value", ""))),
            "function_text": function,
            "go_terms": sorted({x for x in go_terms if x}),
            "ec_terms": [],
        }
        if row["accession"] and row["sequence"]:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote data/uniprot.jsonl")
PY
```

For strict reproduction, use a versioned local JSONL and a split TSV with
columns `accession`, `split` (`train`, `validation`, or `test`), and optional
family/group identifiers. Do not randomly split homologous proteins when
measuring generalization.

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
  --epochs 2 \
  --batch-size 8 \
  --out runs/trajectory_vae.pt
```

Each epoch shuffles the selected records once and traverses them without
replacement. `--steps` is only an optional total-step cap for smoke/debug runs.
Interactive terminals display one `tqdm` batch progress bar per epoch with the
current loss, reconstruction, cosine, KL, and learning rate. Redirected/background
runs automatically keep only the periodic JSON log; pass `--no-progress` to
disable the bar explicitly.
Base VAE training uses an anti-collapse schedule by default: the first 2,000
steps use the posterior mean, sampling noise and KL beta then ramp together over
8,000 steps, and beta stops at 0.001. KL uses 0.01 free nats per latent
dimension, while z4 reconstruction combines embedding MSE with a 0.01-weighted
cosine loss. Checkpoint history records raw/effective KL, active latent units,
posterior scale, noise scale, and cosine reconstruction so collapse remains
observable. These values can be adjusted with `--deterministic-warmup-steps`,
`--noise-ramp-steps`, `--beta-max`, `--kl-free-bits`, and `--cosine-weight`.
The checkpoint contains the VAE weights and the ESM-C model name, but not ESM-C
weights. The first run downloads ESM-C through the installed ESM package and
requires a CUDA GPU for practical training. A small smoke run is:

```bash
prot-lst-train --data data/uniprot.jsonl --limit 32 --steps 5 \
  --batch-size 2 --device cpu --out runs/smoke_vae.pt
```

## Infer A Trajectory

```bash
prot-lst-infer \
  --sequence MKT... \
  --checkpoint runs/trajectory_vae.pt \
  --output runs/example_trajectory.pt
```

The output stores one CPU tensor per protein with shape `[L,4,256]`.

For a FASTA file, use `--fasta` instead of `--sequence`:

```bash
prot-lst-infer --fasta data/example.fasta \
  --checkpoint runs/trajectory_vae.pt \
  --output runs/example_trajectories.pt \
  --device cuda:0
```

Inspect the result:

```bash
python - <<'PY'
import torch
x = torch.load("runs/example_trajectories.pt", map_location="cpu")
for row in x["records"]:
    print(row["name"], tuple(row["trajectory"].shape))
PY
```

## Attribution Training

The attribution workflow requires a deterministic manifest and frozen text
embeddings. If no text teacher is available, train and infer the VAE alone as
above. With a prepared manifest and Function text cache, run:

### Build The Text Cache and continue training

Download a local Qwen3-Embedding model (or another Transformers encoder with a
compatible hidden-state interface), then cache the Function text view. The
cache contains one normalized Function vector per protein; GO and EC are not
training heads and are not injected into the protein encoder.

```bash
prot-lst-cache-text \
  --model /path/to/Qwen3-Embedding-0.6B \
  --data data/uniprot.jsonl \
  --split-file data/splits.tsv \
  --split all \
  --batch-size 8 \
  --max-length 512 \
  --device cuda:0 \
  --out runs/stage2_55k_function_text.pt
```

The model is frozen during cache generation. The standalone script is included at
`prot_lst/scripts/vae_trajectory/cache_text_embeddings.py`; its companion
`prot_lst/text_embedding_models.py` is the only text-encoder dependency.

The locked second-stage experiment first creates a deterministic 45k training,
5k validation, and 5k test split. Training records are selected from the first
stage pool; held-out records come after that pool. Exact sequence and exact
Function-text duplicates cannot cross splits. The TSV is parsed by header name,
not column position:

```bash
prot-lst-build-stage2-split \
  --data data/uniprot_reviewed.jsonl \
  --out data/stage2_55k_splits.tsv \
  --report data/stage2_55k_split_report.json
```

For the current ESM-C 300M/Qwen3 run, the complete resumable preparation and
four-epoch training pipeline is:

```bash
prot-lst-stage2-pipeline \
  --text-model C:/path/to/Qwen3-Embedding-0.6B \
  --device cuda:0
```

It validates all artifact counts, caches one Function view for all 55k records,
creates BF16 ESM-C 300M shards, and trains only `joint_multitask`. Every epoch
visits all 45k training records once with a fresh shuffle and saves
`joint_multitask_4ep.pt.epochN`. The main `joint_multitask_4ep.pt` is the best
validation checkpoint. Epoch checkpoints contain optimizer, contrastive queue,
and RNG state and can be resumed with `--resume`. The locked test split is
cached but never evaluated by this pipeline.

H1 and H2 supervision is confidence-masked independently: a protein with no
annotation for that hierarchy contributes no BCE loss for that head. Within an
annotated protein, unlabelled valid residues are negative positions. Class
weights are fixed statistics from the same 45k supervised subset. Validation
reports Function correlation and Top-1 retrieval plus per-class, Macro, and
Micro AUPRC for H1/H2.

```bash
prot-lst-build-manifest \
  --data data/uniprot.jsonl \
  --split-file data/splits.tsv \
  --text-cache runs/function_text_cache.pt \
  --out runs/attribution_manifest.jsonl

prot-lst-cache-esm \
  --manifest runs/attribution_manifest.jsonl \
  --out-dir runs/esm_shards \
  --esm-model esmc_600m \
  --shard-size 256 --batch-size 4 --device cuda:0
```

Then train one arm and evaluate it:

```bash
prot-lst-train-attribution \
  --manifest runs/attribution_manifest.jsonl \
  --shard-index runs/esm_shards/index.json \
  --text-cache runs/function_text_cache.pt \
  --vae runs/trajectory_vae.pt \
  --arm joint_multitask \
  --out runs/joint_multitask.seed17.pt \
  --epochs 3 --batch-size 16 --model-dim 512 \
  --layers 2 --heads 8 --device cuda:0

prot-lst-evaluate \
  --checkpoint runs/joint_multitask.seed17.pt \
  --manifest runs/attribution_manifest.jsonl \
  --shard-index runs/esm_shards/index.json \
  --text-cache runs/function_text_cache.pt \
  --split validation --out runs/joint_multitask.validation.json \
  --device cuda:0
```

The six supported arms are `esm`, `vae_z3`,
`z3_transformer`, `trajectory_frozen`, `joint_function`, and
`joint_multitask`. Validation selects checkpoints; test evaluation should only
be run after the configuration is locked.

Frozen Qwen3-Embedding vectors are Function targets only. They are not protein
inputs, and GO/EC target labels must not be injected into the encoder path.
