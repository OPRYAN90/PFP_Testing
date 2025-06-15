# 🧬 Unified Protein Function Prediction - Configuration Guide

## 🎯 Philosophy: Everything in One Place

This setup keeps **all related configurations together** for easier tracking and management. The `protein.yaml` files contain model architecture, optimizer, scheduler, and training parameters all in one place.

## 🚀 Quick Start - Running Experiments

### 1. **Default Setup** (Molecular Function prediction)
```bash
python src/train.py
```
This uses:
- Task: MF (Molecular Function, 489 classes)  
- Batch size: 4
- Learning rate: 1e-4
- Model: 640-dim with 2 cross-attention layers

### 2. **Switch to Different Tasks**

#### **Biological Process (BP) Prediction**
```bash
# Option 1: Edit configs/data/protein.yaml 
# Change: task_type: "bp" and batch_size: 2

# Option 2: Command line override
python src/train.py data.task_type=bp data.batch_size=2 model.optimizer.lr=8e-5
```

#### **Cellular Component (CC) Prediction**  
```bash
# Option 1: Edit configs/data/protein.yaml
# Change: task_type: "cc" and batch_size: 6

# Option 2: Command line override  
python src/train.py data.task_type=cc data.batch_size=6 model.optimizer.lr=1.2e-4
```

### 3. **Use Existing Experiment Configs**
```bash
python src/train.py experiment=protein_mf    # Molecular Function
python src/train.py experiment=protein_bp    # Biological Process  
python src/train.py experiment=protein_cc    # Cellular Component
```

## 📁 Unified Configuration Structure

### **Main Configs** (Everything in One Place)
```
configs/
├── train.yaml              # Main entry point
├── data/protein.yaml       # Data config with task switching
├── model/protein.yaml      # Model + Optimizer + Scheduler (unified)
└── experiment/
    ├── protein_mf.yaml     # MF-specific overrides
    ├── protein_bp.yaml     # BP-specific overrides  
    └── protein_cc.yaml     # CC-specific overrides
```

### **Key Configuration Files**

#### **`configs/data/protein.yaml`** - Task Control Center
```yaml
task_type: "mf"           # Switch between "mf", "bp", "cc"
batch_size: 4             # Adjust per task (MF:4, BP:2, CC:6)
data_dir: ${paths.data_dir}/protein_data_pdb
num_workers: 4
train_split: 0.8
```

#### **`configs/model/protein.yaml`** - Everything Model-Related
```yaml  
# Architecture
d_model: 640
n_cross_layers: 2
n_heads: 8

# Optimizer (in same file!)
optimizer:
  lr: 1e-4                # MF:1e-4, BP:8e-5, CC:1.2e-4
  weight_decay: 1e-2

# Scheduler (in same file!)  
scheduler:
  T_max: 100              # Match trainer max_epochs
  eta_min: 1e-6
```

## 🎛️ Task-Specific Recommendations

| Task | Classes | Batch Size | Learning Rate | Label Smoothing | Notes |
|------|---------|------------|---------------|-----------------|-------|
| **MF** | 489 | 4-6 | 1e-4 | 0.02-0.05 | Moderate complexity |
| **BP** | 1943 | 2-3 | 8e-5 | 0.01-0.03 | Large label space, needs more memory |
| **CC** | 320 | 6-8 | 1.2e-4 | 0.03-0.06 | Smaller label space, can use larger batches |

## 🔧 Common Customizations

### **Change Model Architecture**
```bash
# Larger model
python src/train.py model.d_model=1024 model.n_heads=16

# More cross-attention layers  
python src/train.py model.n_cross_layers=4

# Different dropout
python src/train.py model.dropout=0.15
```

### **Adjust Training Parameters**
```bash
# Different learning rate
python src/train.py model.optimizer.lr=5e-4

# Change scheduler max epochs (match your trainer max_epochs)
python src/train.py model.scheduler.T_max=200 trainer.max_epochs=200

# Add label smoothing
python src/train.py model.label_smoothing=0.05
```

### **Alternative Scheduler**
Edit `configs/model/protein.yaml` and uncomment the ReduceLROnPlateau section:
```yaml
# Comment out CosineAnnealingLR and uncomment:
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: "min"
  factor: 0.5
  patience: 10
```

### **Production Mode** (Enable Compilation)
```bash
python src/train.py model.debugging=false
```

## 🧪 Experiment Workflow

### **1. Quick Task Switching**
```bash
# Test MF task
python src/train.py data.task_type=mf data.batch_size=4

# Switch to BP task  
python src/train.py data.task_type=bp data.batch_size=2 model.optimizer.lr=8e-5

# Switch to CC task
python src/train.py data.task_type=cc data.batch_size=6 model.optimizer.lr=1.2e-4
```

### **2. Architecture Exploration**
```bash
# Try different model sizes
python src/train.py model.d_model=512   # Smaller
python src/train.py model.d_model=1024  # Larger

# Try different attention patterns
python src/train.py model.n_heads=4     # Fewer heads
python src/train.py model.n_heads=16    # More heads
```

### **3. Hyperparameter Sweeps**
```bash
# Learning rate sweep
python src/train.py -m model.optimizer.lr=5e-5,1e-4,2e-4,5e-4

# Model dimension sweep
python src/train.py -m model.d_model=256,512,640,1024

# Combined sweep
python src/train.py -m model.d_model=512,640 model.optimizer.lr=1e-4,5e-5
```

## 📊 Monitoring

### **Enable Logging**
```bash
# Weights & Biases
python src/train.py logger=wandb

# TensorBoard  
python src/train.py logger=tensorboard

# With custom experiment name
python src/train.py logger=wandb logger.wandb.name="my_protein_experiment"
```

### **Key Metrics to Track**
- `train/loss`, `val/loss` - Training progress
- `train/auroc`, `val/auroc` - Model performance  
- `val/auroc_best` - Best validation performance
- `val/ap` - Average Precision (multi-label)

## 🔍 Configuration Inspection  

### **View Final Config**
```bash
# See what will actually be used
python src/train.py --cfg job --resolve

# Check specific experiment
python src/train.py experiment=protein_mf --cfg job
```

### **Quick Config Test**
```bash
# Test config loading without training
python src/train.py trainer.fast_dev_run=1
```

## 🛠️ Creating Custom Experiments

Create `configs/experiment/my_custom.yaml`:
```yaml
# @package _global_

defaults:
  - override /data: protein
  - override /model: protein

tags: ["custom", "experiment"]

# Override data settings
data:
  task_type: "mf"
  batch_size: 6

# Override model settings (everything in one place!)
model:
  d_model: 1024
  n_heads: 16
  optimizer:
    lr: 2e-4
    weight_decay: 5e-3
  scheduler:
    T_max: 300

# Override trainer settings
trainer:
  max_epochs: 300
  gradient_clip_val: 2.0
```

Then run: `python src/train.py experiment=my_custom`

## 💡 Pro Tips

1. **Keep It Simple**: Everything related to the model (architecture, optimizer, scheduler) is in `model/protein.yaml`

2. **Task Switching**: Just change `data.task_type` and adjust `batch_size` accordingly

3. **Experimentation**: Use command line overrides for quick tests, create experiment configs for reproducible runs

4. **Memory Management**: 
   - BP task needs smaller batches (2-3)
   - CC task can handle larger batches (6-8)
   - MF task is in between (4-6)

5. **Learning Rates**: Different tasks need different LRs
   - MF: 1e-4 (baseline)
   - BP: 8e-5 (lower for large label space)  
   - CC: 1.2e-4 (slightly higher for small label space) 