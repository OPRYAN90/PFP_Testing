# Unified Protein Function Prediction (Simple README)

A concise guide to train and evaluate the PLM‑fusion model for protein function prediction (GO: MF/BP/CC). This README distills the essentials from the longer paper draft and the codebase.

## What this project does
- **Unified model**: Single architecture for GO term prediction (MF, BP, CC). Easily extensible to localization, EC, stability.
- **PLM‑centric fusion**: Fuses per‑residue embeddings from multiple protein language models via cross‑attention and gated pooling.
- **Sequence‑only**: No MSAs or 3D structures required.

## Repo layout (relevant bits)
- `src/train.py` — Hydra entrypoint for training
- `src/eval.py` — Hydra entrypoint for evaluation
- `configs/` — Hydra configs (data/model/trainer/logger)
- `src/data/protein_datamodule.py` — data module and dataset
- `src/models/` — model code (fusion backbone + heads)

## Environment setup
Option A: pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Option B: conda (if you prefer)
```bash
conda env create -f environment.yaml
conda activate pfp
pip install -e .
```

## Data layout
Point `configs/data/protein.yaml:data_dir` to your PDBCH base directory. Expected structure:
```
<data_dir>/
  train_pdbch/
    <protein_id>/
      sequence.txt
      L.csv
      esmc_emb.pt
      ankh_emb_xl.pt
      prot_t5_emb.pt
      pglm_emb.pt
      mf_go.txt | bp_go.txt | cc_go.txt  # depends on task_type
  val_pdbch/
    ... (same structure)
  test_pdbch/
    ... (same structure)

# In the parent dir of data_dir (one level up):
<data_dir>/..
  nrPDB-GO_2019.06.18_annot.tsv   # GO mapping
  ic_count.pkl                     # dict with keys 'mf','bp','cc' (counts)
```
Notes:
- `ic_count.pkl` is required to compute information content for Smin; the code expects it one directory above `data_dir`.
- All four embedding tensors must have shape `[1, L+2, D]` (BOS/EOS included); they’re squeezed internally.

## Choose task and trainer settings
Key defaults:
- `configs/data/protein.yaml` — set `task_type: "mf" | "bp" | "cc"` and `data_dir`.
- `configs/model/protein.yaml` — inherits `task_type` and sets model dims and optimizer.
- `configs/trainer/protein.yaml` — mixed precision, epochs, devices, etc.
- `configs/logger/wandb.yaml` — set project/entity or disable via `logger=null`.

Override at runtime with Hydra:
```bash
# Example: train CC with a custom data_dir and batch size
python src/train.py data=protein data.task_type=cc data.data_dir=/abs/path/to/PDBCH data.batch_size=32
```

## Train
```bash
python src/train.py \
  data=protein \
  model=protein \
  trainer=protein \
  logger=wandb \
  data.task_type=cc \
  data.data_dir=/abs/path/to/PDBCH
```
- Checkpoints are saved to `logs/` under a Hydra‑generated run dir.
- Early stopping and checkpointing are configured in `configs/callbacks/`.

## Evaluate
Provide a checkpoint path and matching data/model configs:
```bash
python src/eval.py \
  data=protein \
  model=protein \
  trainer=protein \
  data.task_type=cc \
  data.data_dir=/abs/path/to/PDBCH \
  ckpt_path=/abs/path/to/checkpoint.ckpt
```

## Common overrides
- Disable testing after train: `test=false`
- Change precision: `trainer.precision=32-true`
- CPU only: `trainer.accelerator=cpu devices=1`
- Change W&B project: `logger=wandb logger.wandb.project=my_proj`
- Turn off logging: `logger=null`

## Minimal example commands
```bash
# CC task quickstart (adjust data_dir)
python src/train.py data=protein model=protein trainer=protein data.task_type=cc data.data_dir=/data/PDBCH

# MF task
python src/train.py data=protein model=protein trainer=protein data.task_type=mf data.data_dir=/data/PDBCH

# Evaluate a saved checkpoint
python src/eval.py data=protein model=protein trainer=protein data.task_type=cc data.data_dir=/data/PDBCH ckpt_path=/runs/last.ckpt
```

## Citation
- Placeholder citation for the unified PLM‑fusion paper (to be updated).
