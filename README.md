# GR-MTL

Standalone baseline and graph-refined multi-task learning for fundus images.
The repository intentionally contains code and configuration only; dataset files,
image caches, checkpoints, and experiment workbooks are excluded from Git.

## Files

- `train_baseline_mtl.py`: baseline multi-task model.
- `train_graph_refine_mtl.py`: baseline plus Clinical GCN auxiliary refinement.
- `export_test_threshold_metrics.py`: test threshold and auxiliary metrics export.
- `append_experiment_summary.py`: append one run to `experiment_summary.xlsx`.
- `scripts/run_seed42_baseline_rgonly_comparison.ps1`: run the four seed-42
  comparison conditions used in the short paper.

## Model

- A shared fundus-image encoder produces the visual representation.
- Ten task-specific adapters produce latent node features and initial logits.
- A fixed one-layer residual GCN refines the node features.
- Initial and refined auxiliary predictions are both directly supervised.
- The GCN output is not passed to a referral-glaucoma branch.
- The RG head remains as shared-task regularization and reads only the shared
  image representation.

Auxiliary findings:

```text
ANRS, ANRI, RNFLDS, RNFLDI, BCLVS, BCLVI, NVT, DH, LD, LC
```

Clinical graph:

```text
Superior: DH <-> BCLVS <-> RNFLDS <-> ANRS
Inferior: DH <-> BCLVI <-> RNFLDI <-> ANRI
Cross:    ANRS <-> ANRI
NVT:      independent self-loop
LD/LC:    self-loop only
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Baseline:

```powershell
python train_baseline_mtl.py `
  --model-name convnext_tiny `
  --csv path\to\JustRAIGS_processed.csv `
  --image-dir path\to\images_or_cache `
  --cache-dir path\to\cache `
  --output-dir checkpoints\baseline_mtl `
  --epochs 15 `
  --batch-size 32 `
  --moe-dim 128
```

```powershell
python train_graph_refine_mtl.py `
  --model-name convnext_tiny `
  --csv path\to\JustRAIGS_processed.csv `
  --image-dir path\to\images_or_cache `
  --cache-dir path\to\cache `
  --output-dir checkpoints\graph_refine_mtl `
  --epochs 15 `
  --batch-size 32 `
  --moe-dim 128 `
  --aux-rg-valid-only
```

The journal loss is used:

```text
L = L_RG + 0.5 * (L_aux(init) + L_aux(ref))
```

Auxiliary supervision follows the journal label policy:

- NRG: every auxiliary target is `0` with `mask=1`.
- RG with an observed auxiliary label: use the label with `mask=1`.
- RG with a missing auxiliary label: store `0` with `mask=0`, so it is excluded
  from auxiliary loss and metrics.

The best checkpoint is selected by refined auxiliary mean AUROC. The RG head is
optimized jointly, but graph-derived node features never enter or modify its
prediction.

## Seed-42 Comparison

```powershell
.\scripts\run_seed42_baseline_rgonly_comparison.ps1
```

This runs baseline MTL, RG-only frozen-backbone, RG-only unfrozen-backbone,
and RG-only from-scratch conditions with ConvNeXt-Tiny, `moe_dim=128`,
batch size 32, and seed 42. Results are written to the configured output
directory and summarized in `experiment_summary.xlsx`.
