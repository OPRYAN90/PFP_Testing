from typing import Any, Dict, Optional, Tuple, List
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import torch.nn.functional as F


class ProteinDataset(Dataset):
    """Custom Dataset for multi-modal protein function prediction."""
    
    def __init__(self, data_dir: str, protein_ids: List[str], task_type: str = "mf"):
        """
        :param data_dir: Path to protein data directory
        :param protein_ids: List of protein IDs to include
        :param task_type: Which GO task(s) to load labels for ("mf", "bp", "cc")
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.task_type = task_type
        
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        protein_dir = self.data_dir / protein_id
        
        # Load sequence
        with open(protein_dir / "sequence.txt", "r") as f:
            sequence = f.read().strip()
            
        # Load pre-computed embeddings
        msa_data = torch.load(protein_dir / f"{protein_id}_msa_emb.pt")
        esmc_data = torch.load(protein_dir / f"{protein_id}_esmc_emb.pt")
        
        # Extract embeddings from data structures
        # ESM-C: Direct tensor (1, L+2, 1152) -> squeeze to (L+2, 1152) #TODO: CHECK
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor"
        esmc_emb = esmc_data.squeeze(0) 
        
        # MSA: Dictionary with 'embeddings' key (N_seq, L+1, 768)
        if isinstance(msa_data, dict) and 'embeddings' in msa_data: #NOTE: MSA includes multiple sequence alignments in text file 
            msa_emb = msa_data['embeddings']
        else:
            raise ValueError("MSA data should be a dictionary with 'embeddings' key")
        
        # Load labels based on task type
        labels = torch.load(protein_dir / f"{self.task_type}_labels.pt")
            
        # Load protein length info
        length_df = pd.read_csv(protein_dir / "L.csv")
        protein_length = int(length_df.iloc[0, 0])
        
        assert protein_length == esmc_emb.size(0)-2 == msa_emb.size(1)-1, "Sequence/MSA/Length mismatch"
        
        return {
            "protein_id": protein_id,
            "sequence": sequence,
            "sequence_emb": esmc_emb,  # Shape: (L+2, d_model) 
            "msa_emb": msa_emb,        # Shape: (N_seq, L+1, d_msa)
            "labels": labels,
            "length": protein_length
        }


class ProteinDataModule(LightningDataModule):
    """DataModule for protein function prediction.
    
    Handles two input modalities:
    1. ESM-C embeddings (pre-computed)
    2. MSA embeddings (pre-computed)
    
    Automatically scans dataset to find proteins with both embedding types.
    """

    def __init__(
        self,
        data_dir: str = "data/protein_data_pdb",
        task_type: str = "mf",  # "mf", "bp", or "cc"
        batch_size: int = 4,    # Updated to 4 as requested
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.8,  # Split valid proteins into train/val
        # max_sequence_length: int = 1024,  # For padding
    ) -> None:
        """Initialize ProteinDataModule.

        :param data_dir: Path to protein data directory
        :param task_type: Which GO tasks to train on ("mf", "bp", "cc")
        :param batch_size: Batch size for training
        :param num_workers: Number of data loading workers
        :param pin_memory: Whether to pin memory for GPU transfer
        :param train_split: Fraction of valid proteins to use for training
        :param max_sequence_length: Maximum sequence length for padding
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.valid_protein_ids: List[str] = []

    @property
    def num_classes(self) -> int:
        """Get number of classes for current task."""
        if self.hparams.task_type == "mf":
            return 489
        elif self.hparams.task_type == "bp":
            return 1943
        elif self.hparams.task_type == "cc":
            return 320
        else:
            raise ValueError(f"Unknown task_type: {self.hparams.task_type}")

    def _find_valid_proteins(self, data_dir: Path) -> List[str]:
        """Scan dataset to find proteins with both ESM-C and MSA embeddings."""
        valid_proteins = []
        
        print(f"Scanning {data_dir} for proteins with required embeddings...")
        
        for protein_dir in data_dir.iterdir():
            if not protein_dir.is_dir():
                continue
                
            protein_id = protein_dir.name
            required_files = [
                f"{protein_id}_esmc_emb.pt",
                f"{protein_id}_msa_emb.pt",
                "sequence.txt",
                f"{self.hparams.task_type}_labels.pt",
                "L.csv"
            ] #TODO: Consider proteins that may not have repsective label file 
            
            # Check if all required files exist
            if all((protein_dir / file).exists() for file in required_files):
                valid_proteins.append(protein_id)
                print(f"✓ Found valid protein: {protein_id}")
        
        print(f"Found {len(valid_proteins)} valid proteins with all required files")
        return valid_proteins

    def prepare_data(self) -> None:
        """Find all proteins with required embeddings."""
        # ✅ Skip if already done
        if self.valid_protein_ids:
            print(f"Using previously found {len(self.valid_protein_ids)} valid proteins")
            return
        
        data_dir = Path(self.hparams.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Find all valid proteins (only once!)
        self.valid_protein_ids = self._find_valid_proteins(data_dir)
        
        if len(self.valid_protein_ids) < 2:
            raise ValueError(f"Need at least 2 valid proteins for train/val split, found {len(self.valid_protein_ids)}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Split valid proteins into train/val sets."""
        # FIX: Guard world_size access for unit tests
        if hasattr(self.trainer, 'world_size'):
            # Divide batch size by number of devices
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        else:
            # Fallback for unit tests or manual setup
            self.batch_size_per_device = self.hparams.batch_size

        # Load datasets only if not already loaded
        if not self.data_train and not self.data_val:
            
            # Split valid proteins into train/val
            n_train = int(len(self.valid_protein_ids) * self.hparams.train_split)
            train_protein_ids = self.valid_protein_ids[:n_train]
            val_protein_ids = self.valid_protein_ids[n_train:]
            
            # Ensure we have at least 1 protein for validation
            if len(val_protein_ids) == 0:
                val_protein_ids = [train_protein_ids[-1]]  # Use last training protein for val
            
            print(f"Training proteins ({len(train_protein_ids)}): {train_protein_ids}")
            print(f"Validation proteins ({len(val_protein_ids)}): {val_protein_ids}")
            
            # Create datasets
            self.data_train = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=train_protein_ids,
                task_type=self.hparams.task_type
            )
            
            self.data_val = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=val_protein_ids,
                task_type=self.hparams.task_type
            )
            
            # Use validation set as test set for now
            self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader[Any]:
        """Create training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences and matrices."""
        if len(batch) == 1:
            return batch[0]
        
        # Handle batching for multiple samples
        protein_ids = [item["protein_id"] for item in batch]
        sequences = [item["sequence"] for item in batch]
        lengths = [item["length"] for item in batch]        # residue counts


        # +2  (CLS + EOS)  — every sequence tensor on disk already has them
        # +2  (CLS + EOS)  — every MSA row on disk has CLS; EOS is added later
        max_len_seq = max(lengths) + 2        # unchanged (CLS+res+EOS)
        assert max_len_seq <= 1024, "Sequence length must be less than or equal to 1024"
        
        # Pad sequence embeddings  [max_len_seq, d_model]
        seq_embs = []
        for item in batch:
            seq_emb = item["sequence_emb"]  # [L, d_model]
            if seq_emb.size(0) < max_len_seq:
                pad_size = max_len_seq - seq_emb.size(0)
                seq_emb = F.pad(seq_emb, (0, 0, 0, pad_size), value=0)
            seq_embs.append(seq_emb)
        
        # Pad MSA embeddings  [N_seq, max_len_seq, d_msa]
        msa_embs = []
        for item in batch:
            msa_emb = item["msa_emb"]  # [N_seq, L, d_msa]
            if msa_emb.size(1) < max_len_seq:     # pad so every sample has room for EOS
                pad_size = max_len_seq - msa_emb.size(1)
                # FIX: Proper 6-value padding tuple for 3D tensor [N_seq, L, d_msa]
                msa_emb = F.pad(msa_emb, (
                    0, 0,           # d_msa dimension (no padding)
                    0, pad_size,    # L dimension (pad at end)  
                    0, 0            # N_seq dimension (no padding)
                ), value=0)
            msa_embs.append(msa_emb)
        
        # Stack embeddings
        sequence_emb = torch.stack(seq_embs)    # [B, L_seq_max, d_model]
        msa_emb      = torch.stack(msa_embs)    # [B, N_seq, L_msa_max, d_msa]

        #TODO: FORCE MSA LENGTHS TO BE THE SAME AS SEQUENCE LENGTHS
        
        assert sequence_emb.size(1) == msa_emb.size(2), "Sequence and MSA lengths must be the same"

        # Stack labels
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        
        # Masking boolean tensor [B, max_len_seq]
        pad_mask = torch.tensor(
            [[i >= l + 2 for i in range(max_len_seq)] for l in lengths],
            dtype=torch.bool
        )
        
        # FIX 2: Build lengths on CPU, let Lightning move it once
        lengths = torch.tensor(lengths)  # ← FIX: No device specified = CPU
        
        return {
            "protein_id": protein_ids,
            "sequence": sequences,
            "sequence_emb": sequence_emb,
            "msa_emb": msa_emb,
            "labels": labels,
            "pad_mask": pad_mask,              
            "lengths": lengths,
        }

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up resources."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {"valid_protein_ids": self.valid_protein_ids}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self.valid_protein_ids = state_dict.get("valid_protein_ids", [])
