# Omni-Prot: Unifying Protein Function Prediction with PLM-Centric Fusion

Omni-Prot is a machine learning model for protein-level prediction using fused embeddings from multiple protein language models (PLMs).

The primary task in this repository is Gene Ontology (GO) function prediction, with EC classification and regression included as lightweight variants of the same pipeline.

The core idea employs precompute protein embeddings from multiple PLMs into the Omni-Prot architecture which returns predictions that can be fine-tuned for a variety of specefic protein tasks.

---

## Project Structure

### Entrypoints

- `src/train.py` — main training entrypoint
- `src/eval.py` — evaluation entrypoint

### Main task: GO prediction

- `src/data/protein_datamodule.py` — loads GO datasets and protein embeddings
- `src/models/protein_module.py` — main LightningModule for multi-label GO prediction
- `src/data/go_utils.py` — GO utilities and evaluation helpers (propagation, AUPR, CAFA Fmax, Smin)

### Other task variants

These reuse the same structure as GO, with different labels/metrics:

- **EC prediction:** `src/data/EC_datamodule.py`, `src/models/EC_module.py`
- **Regression:** `src/data/regression_datamodule.py`, `src/models/regression_module.py`

### Callbacks

- `src/callbacks/ema.py` — exponential moving average weights
- `src/callbacks/perf.py` — performance monitoring
- `src/callbacks/weight_averaging.py` — weight averaging callback

### Utilities

- `src/utils/lr_schedulers.py` — learning rate scheduler helpers, including cosine annealing with warmup
- `src/utils/instantiators.py` — Hydra-based instantiation for callbacks/loggers

### Experimental

- `src/models/egnn.py` — an example of an experiment run with the EGNN architecture.

Note: Many other experiments were run and tracked on Weights & Biases. These experiments covered tweaks including architectural variations and input variations like MSAs (which often incorporated much architectural variation). Select cases can be noted in the version history, though full explanation may not provided in the history. 

### Notebooks

- `notebooks/GO_Sliding.ipynb` — includes sliding window algorithm to obtain precomputed embeddings from the relevant PLMs

### Configuration

Hydra configs define what to run (data/model/trainer/logger/callbacks):

- `configs/train.yaml`, `configs/eval.yaml` — main entry configs
- `configs/data/` — datamodule configs (`protein.yaml`, `ec.yaml`, `regression.yaml`)
- `configs/model/` — model configs (fusion architecture params, optimizer, scheduler)
- `configs/trainer/` — Lightning Trainer settings (precision, epochs, etc.)
- `configs/callbacks/` — task-specific bundles (checkpointing, early stopping, EMA)
- `configs/logger/` — logging configs (wandb, tensorboard, csv)
- `configs/paths/default.yaml` — defines `PROJECT_ROOT`, `data_dir`, `log_dir`, `output_dir`

---

## Data Layout (GO)

The GO datamodule expects the following structure:

```
data/
├── nrPDB-GO_2019.06.18_annot.tsv
├── ic_count.pkl
└── PDBCH/
    ├── train_pdbch/
    │   └── <protein_id>/
    │       ├── sequence.txt
    │       ├── L.csv
    │       ├── esmc_emb.pt
    │       ├── prot_t5_emb.pt
    │       ├── ankh_emb_xl.pt
    │       ├── pglm_emb.pt
    │       └── mf_go.txt  # or bp_go.txt / cc_go.txt
    ├── val_pdbch/
    │   └── <protein_id>/
    │       └── (same required files)
    └── test_pdbch/
        └── <protein_id>/
            └── (same required files)
```

Each protein directory must contain the required embedding files and the ontology-specific label file. Proteins with missing files or empty label files are skipped.

---

## Training (GO example)

Train the GO model (MF by default):

```bash
python src/train.py data=protein model=protein trainer=protein callbacks=protein
```

Switch ontology:

```bash
python src/train.py data=protein data.task_type=bp
python src/train.py data=protein data.task_type=cc
```

Evaluate from a checkpoint:

```bash
python src/eval.py data=protein model=protein ckpt_path=/path/to/checkpoint.ckpt
```

Hydra controls all task, model, and trainer selection through configuration.
