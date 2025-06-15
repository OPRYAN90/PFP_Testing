# Protein Function Prediction Configuration Guide

This directory contains Hydra configurations for the multi-modal protein function prediction system. The configurations are organized to support different training scenarios and GO tasks.

## Architecture Overview

The system uses a multi-modal architecture with:
- **Sequence Encoder**: Processes pre-computed ESM-C embeddings
- **MSA Encoder**: Processes MSA embeddings with conservation-aware attention
- **Cross-Modal Attention**: Fuses sequence and MSA representations
- **Classification Head**: Multi-label prediction for GO terms

## Configuration Structure

### Model Configurations (`model/`)

- **`protein.yaml`**: Standard development model configuration
  - `d_model: 640` - Base model dimension
  - `d_msa: 768` - MSA embedding dimension  
  - `debugging: true` - Disables torch.compile for easier debugging
  - Conservative optimizer settings

- **`protein_prod.yaml`**: Production model configuration
  - `debugging: false` - Enables torch.compile for speed
  - OneCycleLR scheduler with warmup
  - Higher learning rate and more aggressive settings

### Data Configurations (`data/`)

- **`protein.yaml`**: Base data configuration
- **`protein_mf.yaml`**: Molecular Function (489 classes)
- **`protein_bp.yaml`**: Biological Process (1943 classes)  
- **`protein_cc.yaml`**: Cellular Component (320 classes)

### Experiment Configurations (`experiment/`)

Ready-to-run experiment configurations that combine model, data, trainer, and callbacks:

- **`protein_mf.yaml`**: Molecular Function prediction
- **`protein_bp.yaml`**: Biological Process prediction
- **`protein_cc.yaml`**: Cellular Component prediction
- **`protein_mf_prod.yaml`**: Production MF training with torch.compile

### Trainer Configurations (`trainer/`)

- **`protein.yaml`**: Protein-specific trainer settings
  - Mixed precision training
  - Gradient clipping
  - Longer training epochs
  - Memory-efficient settings

### Callback Configurations (`callbacks/`)

- **`protein.yaml`**: Protein-specific callbacks
  - Monitors `val/auroc` instead of `val/acc`
  - Appropriate patience for slow protein training
  - Saves top-k models based on AUROC

### Debug Configurations (`debug/`)

- **`protein.yaml`**: Protein development debugging
  - Limited batches for fast iteration
  - Anomaly detection enabled
  - Full precision for numerical stability
  - Offline logging

## Usage Examples

### Basic Training

```bash
# Train molecular function prediction
python train.py experiment=protein_mf

# Train biological process prediction  
python train.py experiment=protein_bp

# Train cellular component prediction
python train.py experiment=protein_cc
```

### Production Training

```bash
# Production training with torch.compile
python train.py experiment=protein_mf_prod

# Override specific parameters
python train.py experiment=protein_mf_prod trainer.max_epochs=500 model.optimizer.lr=3e-4
```

### Development & Debugging

```bash
# Quick debugging run
python train.py experiment=protein_mf debug=protein

# Test with limited data
python train.py experiment=protein_mf trainer.limit_train_batches=10 trainer.max_epochs=2

# Profile performance
python train.py experiment=protein_mf debug=profiler
```

### Custom Configurations

```bash
# Mix and match components
python train.py data=protein_mf model=protein_prod trainer=protein callbacks=protein

# Override individual parameters
python train.py experiment=protein_mf model.d_model=512 data.batch_size=2
```

## Key Parameters

### Model Architecture
- `d_model`: Base embedding dimension (default: 640)
- `d_msa`: MSA embedding dimension (default: 768)
- `n_seq_layers`: Additional sequence transformer layers (default: 2)
- `n_cross_layers`: Cross-attention layers (default: 2)
- `n_heads`: Attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

### Training Settings
- `batch_size`: Currently limited to 1 due to variable sequence lengths
- `accumulate_grad_batches`: Effective batch size multiplier
- `gradient_clip_val`: Gradient clipping threshold (default: 1.0)
- `precision`: "16-mixed" for speed, "32-true" for debugging

### Task-Specific Settings
- **MF (489 classes)**: Moderate learning rate, balanced settings
- **BP (1943 classes)**: Lower learning rate, more regularization
- **CC (320 classes)**: Higher learning rate, fewer epochs

## Performance Optimization

### Memory Optimization
- Use `precision: "16-mixed"` for reduced memory usage
- Increase `accumulate_grad_batches` instead of `batch_size`
- Enable gradient checkpointing in model if needed

### Speed Optimization
- Use `debugging: false` to enable torch.compile
- Increase `num_workers` for data loading
- Use `pin_memory: true` for GPU training

### Debugging Tips
- Use `debug=protein` for development
- Set `trainer.detect_anomaly=true` for numerical issues
- Use `precision: "32-true"` for numerical stability
- Set `num_workers: 0` for easier debugging

## Monitoring & Logging

The system logs the following key metrics:
- `train/loss`, `val/loss`, `test/loss`: BCE loss values
- `train/auroc`, `val/auroc`, `test/auroc`: Area under ROC curve
- `val/ap`: Average precision (validation only)
- `val/auroc_best`: Best validation AUROC achieved

Checkpoints are saved based on `val/auroc` with filenames including the AUROC score for easy identification.

## Common Issues & Solutions

### Memory Issues
- Reduce `batch_size` to 1
- Increase `accumulate_grad_batches`
- Use mixed precision: `precision: "16-mixed"`
- Reduce `d_model` if necessary

### Slow Training
- Enable compilation: `debugging: false`
- Increase `num_workers`
- Use `check_val_every_n_epoch: 10` for less frequent validation
- Consider reducing `n_seq_layers` or `n_cross_layers`

### Convergence Issues
- Increase `patience` in early stopping
- Reduce learning rate
- Increase `label_smoothing`
- Check gradient clipping value

### Data Loading Issues
- Verify data directory structure
- Check that all required files exist (sequence.txt, *_msa_emb.pt, *_labels.pt)
- Use `num_workers: 0` for debugging data loading

## Configuration Hierarchy

Hydra resolves configurations in this order (later overrides earlier):
1. Base configurations (model, data, trainer, callbacks)
2. Experiment configuration overrides
3. Command-line overrides

This allows for flexible configuration while maintaining sensible defaults. 