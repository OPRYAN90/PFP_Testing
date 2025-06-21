from typing import Any, Dict, Optional, Tuple, List
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import get_worker_info
import pickle as pkl
from .go_utils import load_go_dict


class ProteinDataset(Dataset):
    """Custom Dataset for multi-modal protein function prediction."""
    
    # ── GLOBAL CACHES (per-process, not per-thread) ────────────────
    _msa_model         = None
    _msa_batch_conv    = None

    # static device decision: *this* process only
    _device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _use_fp16          = ( #NOTE: CHANGE GPU PRECISION
        _device.type == "cuda" and torch.cuda.get_device_capability(_device)[0] >= 7
    )
    
    # class-level cache so every worker only builds it once
    _go_dicts = {}
    
    def __init__(self, data_dir: str, protein_ids: List[str], task_type: str = "mf", split: str = "train"):
        """
        :param data_dir: Path to protein data directory (base PDBCH directory)
        :param protein_ids: List of protein IDs to include
        :param task_type: Which GO task(s) to load labels for ("mf", "bp", "cc")
        :param split: Which data split ("train", "val", "test")
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.task_type = task_type
        self.split = split
        
        # Determine the specific split directory
        self.split_dir = self.data_dir / f"{split}_pdbch"
        
    def __len__(self):
        return len(self.protein_ids)
    
    # --------------------------------------------------------------
    @classmethod
    def _get_msa_model(cls):
        """
        Load the ESM-MSA model once per process.
        """
        
        if cls._msa_model:
            return cls._msa_model, cls._msa_batch_conv

        try:
            import esm
            cls._msa_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            cls._msa_batch_conv = alphabet.get_batch_converter()
            cls._msa_model.eval().to(cls._device)
            if cls._use_fp16:
                cls._msa_model.half()
            if get_worker_info() is None:        # main process only
                print(
                    f"[ESM-MSA] loaded on {cls._device} "
                    f"({'FP16' if cls._use_fp16 else 'FP32'})"
                    f"{'ENSURE Mixed-precision is enabled' if cls._use_fp16 else 'ESSENTIAL: Single-precision is enabled'}"
                )
        except Exception as e:
            raise RuntimeError(f"ESM-MSA load failed: {e}")

        return cls._msa_model, cls._msa_batch_conv

    @staticmethod
    def _parse_a3m(path: Path) -> List[str]:
        """Parse A3M file - first sequence is query, max 256 total sequences."""
        seqs, seq = [], []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                if line[0] == ">":                     # header
                    if seq:
                        seqs.append("".join(seq))
                    seq = []                           # reset
                else:
                    seq.append(line)
            if seq:  # add final sequence
                seqs.append("".join(seq))
        
        assert 1 <= len(seqs) <= 256, f"Expected 1-256 sequences, got {len(seqs)} from {path}"
        return seqs

    @torch.inference_mode()
    def _compute_msa_embeddings(self, a3m_file: Path) -> torch.Tensor:
        model, batch_converter = self._get_msa_model()
        seqs = self._parse_a3m(a3m_file)
        msa = [(f"seq{i}", s) for i, s in enumerate(seqs)]

        _, _, tok = batch_converter([msa])
        tok = tok.to(self._device)
        if self._use_fp16:           # convert input only if model is FP16
            tok = tok.half()

        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16 if self._use_fp16 else torch.float32,
            enabled=self._use_fp16,
        ):
            rep = model(tok, repr_layers=[12])["representations"][12]

        return rep.squeeze(0).cpu()  # (N_seq, L+1, d) TODO: CONSIDER .float() if mixed-precision is not working 

    def _parse_go_labels(self, go_file_path: Path) -> torch.Tensor:
        """
        Convert the GO IDs inside <ontology>_go.txt into a multi-label tensor.
        Empty file  ➜  all-zero vector (negative example).
        Unknown ID ➜  ignored but logged once.
        """
        # 1. fetch / cache mapping ------------------------------------------------
        if self.task_type not in self._go_dicts:
            tsv_path = self.data_dir / "nrPDB-GO_2019.06.18_annot.tsv"
            self._go_dicts[self.task_type] = load_go_dict(tsv_path, self.task_type)
        go2idx = self._go_dicts[self.task_type]
        num_classes = len(go2idx)

        # 2. build empty vector ---------------------------------------------------
        labels = torch.zeros(num_classes, dtype=torch.float32)

        # 3. fill in positive indices -------------------------------------------
        if go_file_path.exists() and go_file_path.stat().st_size > 0:
            with go_file_path.open() as fh:
                for raw in fh:
                    go_id = raw.strip()
                    if not go_id:
                        continue
                    try:
                        labels[go2idx[go_id]] = 1.0
                    except KeyError:
                        print(f"[WARN] {go_id} not in mapping ({self.task_type})")
        return labels
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        protein_dir = self.split_dir / protein_id
        
        # Load sequence
        with open(protein_dir / "sequence.txt", "r") as f:
            sequence = f.read().strip()
            
        # Load pre-computed ESM-C embeddings (required file)
        esmc_file = protein_dir / f"{protein_id}_esmc_emb.pt"
        esmc_data = torch.load(esmc_file)
        # Extract embeddings from data structures
        # ESM-C: Direct tensor (1, L+2, 1152) -> squeeze to (L+2, 1152)
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor"
        esmc_emb = esmc_data.squeeze(0)
        
        # Load MSA data from .a3m file and compute embeddings on-the-fly
        a3m_file = protein_dir / "final_filtered_256_stripped.a3m"
        msa_emb = self._compute_msa_embeddings(a3m_file)
        
        # Load labels based on task type from GO text files
        label_file = protein_dir / f"{self.task_type}_go.txt"
        
            # Parse GO labels from text file format
        labels = self._parse_go_labels(label_file)
            
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
    1. ESM-C embeddings (pre-computed, required as .pt files)
    2. MSA embeddings (computed on-the-fly from .a3m files)
    
    Now supports three predefined splits: train, val, test from separate directories.
    """

    def __init__(
        self,
        data_dir: str = "data/PDBCH",  # Updated to point to PDBCH base directory
        task_type: str = "mf",  # "mf", "bp", or "cc"
        batch_size: int = 4,    # Updated to 4 as requested
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Initialize ProteinDataModule.

        :param data_dir: Path to PDBCH base directory 
        :param task_type: Which GO tasks to train on ("mf", "bp", "cc")
        :param batch_size: Batch size for training
        :param num_workers: Number of data loading workers
        :param pin_memory: Whether to pin memory for GPU transfer
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        # Store protein IDs for each split
        self.train_protein_ids: List[str] = []
        self.val_protein_ids: List[str] = []
        self.test_protein_ids: List[str] = []

    def get_num_classes(self, task_type_: str) -> int:
        """Get number of classes for a given task type with validation.
        
        :param task_type: Task type ("mf", "bp", "cc")
        :return: Number of classes for the task
        """
        expected_counts = {"mf": 489, "bp": 1943, "cc": 320}
        assert task_type_ in expected_counts, f"Unknown task_type: {task_type_}"
        
        tsv = Path(self.hparams.data_dir) / "nrPDB-GO_2019.06.18_annot.tsv"
        go_dict = load_go_dict(tsv, task_type_)
        actual_count = len(go_dict)
        
        assert actual_count == expected_counts[task_type_], \
            f"Expected {expected_counts[task_type_]} classes for {task_type_}, got {actual_count}"
        
        return actual_count

    @property
    def num_classes(self) -> int:
        """Get number of classes for current task."""
        return self.get_num_classes(self.hparams.task_type)

    @property
    def ic_vector(self) -> torch.Tensor:
        """Get the information content vector for Smin computation."""
        return self._ic_vector

    def _find_valid_proteins_in_split(self, split_dir: Path) -> List[str]:
        """Scan a split directory to find proteins with non-empty GO label files."""
        valid_proteins = []
        
        print(f"Scanning {split_dir} for proteins with non-empty {self.hparams.task_type} labels...")
        
        for protein_dir in split_dir.iterdir():
            if not protein_dir.is_dir():
                continue
                
            protein_id = protein_dir.name
            #TODO: CHECK PROTEIN MISMATCH IN HEAL DATASET
            # Check that all required files exist
            required_files = [
                "L.csv",
                "sequence.txt", 
                "final_filtered_256_stripped.a3m",
                f"{protein_id}_esmc_emb.pt",
                f"{self.hparams.task_type}_go.txt"
            ]
            
            missing_files = [f for f in required_files if not (protein_dir / f).exists()]
            
            if missing_files:
                # This should not happen - all valid proteins must have these files
                raise FileNotFoundError(f"Protein {protein_id} missing required files: {missing_files}")
            
            valid_proteins.append(protein_id)
        
        print(f"Found {len(valid_proteins)} valid proteins with {self.hparams.task_type} labels in {split_dir.name}")
        return valid_proteins

    def prepare_data(self) -> None:
        """Find all proteins with required files in each split directory."""
        # Skip if already done
        if self.train_protein_ids and self.val_protein_ids and self.test_protein_ids:
            print(f"Using previously found proteins: "
                  f"train={len(self.train_protein_ids)}, "
                  f"val={len(self.val_protein_ids)}, "
                  f"test={len(self.test_protein_ids)}")
            return
        
        data_dir = Path(self.hparams.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Check for all three split directories
        train_dir = data_dir / "train_pdbch"
        val_dir = data_dir / "val_pdbch"
        test_dir = data_dir / "test_pdbch"
        
        missing_dirs = []
        if not train_dir.exists():
            missing_dirs.append("train_pdbch")
        if not val_dir.exists():
            missing_dirs.append("val_pdbch")
        if not test_dir.exists():
            missing_dirs.append("test_pdbch")
        
        if missing_dirs:
            raise ValueError(f"Missing split directories in {data_dir}: {missing_dirs}")
        
        # Find valid proteins in each split
        self.train_protein_ids = self._find_valid_proteins_in_split(train_dir)
        self.val_protein_ids = self._find_valid_proteins_in_split(val_dir)
        self.test_protein_ids = self._find_valid_proteins_in_split(test_dir)
        
        if len(self.train_protein_ids) == 0:
            raise ValueError("No valid training proteins found")
        if len(self.val_protein_ids) == 0:
            raise ValueError("No valid validation proteins found")
        if len(self.test_protein_ids) == 0:
            raise ValueError("No valid test proteins found")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for each split."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets only if not already loaded
        if not self.data_train and not self.data_val and not self.data_test:
            
            print(f"Training proteins ({len(self.train_protein_ids)}): {self.train_protein_ids[:5]}{'...' if len(self.train_protein_ids) > 5 else ''}")
            print(f"Validation proteins ({len(self.val_protein_ids)}): {self.val_protein_ids[:5]}{'...' if len(self.val_protein_ids) > 5 else ''}")
            print(f"Test proteins ({len(self.test_protein_ids)}): {self.test_protein_ids[:5]}{'...' if len(self.test_protein_ids) > 5 else ''}")
            
            # Create datasets for each split
            self.data_train = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.train_protein_ids,
                task_type=self.hparams.task_type,
                split="train"
            )
            
            # ---------------------------------------------------------
            # HEAL information–content vector – loaded from ic_count.pkl
            # ---------------------------------------------------------
            if not hasattr(self, "_ic_vector"):
                ic_file = Path("data/ic_count.pkl")
                if ic_file.exists():
                    with ic_file.open("rb") as f:
                        ic_count = pkl.load(f)               # dict with keys 'bp','mf','cc'
                    counts = torch.tensor(
                        ic_count[self.hparams.task_type], dtype=torch.float
                    )
                    counts[counts == 0] = 1                  # avoid log(0)
                    # HEAL constant: 69 709 training proteins
                    self._ic_vector = (-torch.log2(counts / 69_709)).float()
                    print(f"✅  IC vector loaded from {ic_file}")
                else:
                    raise FileNotFoundError("ic_count.pkl not found")
            # Verify length agreement
            assert len(self._ic_vector) == self.get_num_classes(self.hparams.task_type)
            
            self.data_val = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.val_protein_ids,
                task_type=self.hparams.task_type,
                split="val"
            )
            
            self.data_test = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.test_protein_ids,
                task_type=self.hparams.task_type,
                split="test"
            )

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
        
        # ── NEW: row-axis padding ────────────────────────────────────
        max_n_seq = max(item["msa_emb"].size(0) for item in batch)

        msa_embs = []
        for item in batch:
            msa = item["msa_emb"]           # [N_seq, L, d_msa]

            # pad L axis (existing code)
            if msa.size(1) < max_len_seq:
                pad_L = max_len_seq - msa.size(1)
                msa = F.pad(msa, (0, 0, 0, pad_L, 0, 0), value=0)

            # pad N_seq axis (row axis) **at the bottom**
            if msa.size(0) < max_n_seq:
                pad_rows = max_n_seq - msa.size(0)
                msa = F.pad(msa, (0, 0, 0, 0, 0, pad_rows), value=0)

            msa_embs.append(msa)
        
        # Stack embeddings
        sequence_emb = torch.stack(seq_embs)    # [B, L_seq_max, d_model]
        msa_emb      = torch.stack(msa_embs)    # [B, N_seq, L_msa_max, d_msa]


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
        return {
            "train_protein_ids": self.train_protein_ids,
            "val_protein_ids": self.val_protein_ids,
            "test_protein_ids": self.test_protein_ids,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self.train_protein_ids = state_dict.get("train_protein_ids", [])
        self.val_protein_ids = state_dict.get("val_protein_ids", [])
        self.test_protein_ids = state_dict.get("test_protein_ids", [])