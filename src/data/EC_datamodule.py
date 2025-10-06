from typing import Any, Dict, Optional, List
import random
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch.nn.functional as F
import warnings
import pickle
EC_IDX_PKL = Path("/teamspace/studios/this_studio/PFP_Testing/data/EC_idx.pkl")
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


class ProteinDataset(Dataset):
    """Sequence-only EC dataset using precomputed PLM embeddings + EC labels."""

    # (Avoid using a class-level cache for multiprocessing safety.)

    def __init__(self, data_dir: str, protein_ids: List[str], split: str = "train", ec2idx: Optional[Dict[str, int]] = None):
        """
        :param data_dir: Path to split directory (e.g., EC_Number/Training_set)
        :param protein_ids: List of protein IDs to include
        :param split: Which data split ("train", "val", "test")
        :param ec2idx: Optional mapping from EC leaf to class index
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.split = split
        # Store mapping on the *instance* so it is pickled to DataLoader workers.
        self.ec2idx: Dict[str, int] = dict(ec2idx or {})
        
    def __len__(self):
        return len(self.protein_ids)


    def _parse_ec_labels(self, ec_file_path: Path) -> torch.Tensor:
        """Parse EC leaves from ec_numbers.txt into multi-hot over ec2idx.

        Unknown or partial codes (containing '-') are ignored. All-zero vectors are allowed.
        """
        labels = torch.zeros(len(self.ec2idx), dtype=torch.float32)
        if ec_file_path.exists():
            for raw in ec_file_path.read_text().splitlines():
                ec_code = raw.strip()
                if not ec_code or "-" in ec_code:
                    print(f"[WARN] {ec_code} is not a valid EC code")
                    continue
                idx = self.ec2idx.get(ec_code)
                if idx is not None:
                    labels[idx] = 1.0
                elif idx is None:
                    print(f"[WARN] {ec_code} not in mapping")
        assert labels.sum() > 0, f"All-zero label vector for {ec_file_path} (no valid EC codes found)"
        return labels
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        protein_dir = self.data_dir / protein_id
        
        # Load sequence
        with open(protein_dir / "sequence.txt", "r") as f:
            sequence = f.read().strip()
            
        # Load pre-computed ESM-C embeddings (required file)
        esmc_file = protein_dir / "esmc_emb.pt"
        esmc_data = torch.load(esmc_file)
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor"
        esmc_emb = esmc_data.squeeze(0)
        
        # Load pre-computed Ankh3-XLarge embeddings (required file)
        ankh_file = protein_dir / "ankh_emb_xl.pt"
        ankh_data = torch.load(ankh_file)
        # Extract embeddings from data structures
        # Ankh3-XLarge: Direct tensor (1, L+2, d_ankh) -> squeeze to (L+2, d_ankh)
        assert ankh_data.dim() == 3, "Ankh3-XLarge data should be a 3D tensor"
        ankh_emb = ankh_data.squeeze(0)

        # Load pre-computed ProtT5 embeddings (required file)
        prot_file = protein_dir / "prot_t5_emb.pt"
        prot_data = torch.load(prot_file)
        assert prot_data.dim() == 3, "ProtT5 data should be a 3D tensor"
        prot_emb = prot_data.squeeze(0)

        # Load pre-computed XTrimoPGLM embeddings (required file)
        pglm_file = protein_dir / "pglm_emb.pt"
        pglm_data = torch.load(pglm_file)
        assert pglm_data.dim() == 3, "PGLM data should be a 3D tensor"
        pglm_emb = pglm_data.squeeze(0)

        # Ensure all embeddings have the same sequence length
        assert esmc_emb.size(0) == ankh_emb.size(0) == prot_emb.size(0) == pglm_emb.size(0), \
            f"Embeddings have different lengths: ESM-C={esmc_emb.size(0)}, Ankh={ankh_emb.size(0)}, ProtT5={prot_emb.size(0)}, PGLM={pglm_emb.size(0)}"
            
        # Ensure that BOS/EOS token embeddings (positions 0 and L+1) are zeroed out for all models
        for emb in (esmc_emb, ankh_emb, prot_emb, pglm_emb):
            emb[0] = 0.0
            emb[-1] = 0.0

        # EC labels
        labels = self._parse_ec_labels(protein_dir / "ec_numbers.txt")

        # Protein length from sequence (no L.csv in EC layout)
        protein_length = len(sequence)
        
        sample = {
            "protein_id": protein_id,
            "sequence": sequence,
            "esmc_emb": esmc_emb,      # Shape: (L+2, d_esmc)
            "ankh_emb": ankh_emb,      # Shape: (L+2, d_ankh)
            "prot_emb": prot_emb,      # Shape: (L+2, d_prot)
            "pglm_emb": pglm_emb,      # Shape: (L+2, d_pglm)
            "length": protein_length,
            "labels": labels,
        }

        return sample


class ProteinDataModule(LightningDataModule):
    """EC Number DataModule (sequence-only, PLM embeddings)."""

    # Help static type checkers recognise dynamically-added Hydra attribute
    hparams: Any

    def __init__(
        self,
        data_dir: str = "data/EC_Number",
        test_split_name: str = "Price-149",
        val_split_name: str = "NEW-392",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the EC data module.

        :param data_dir: Path to EC_Number root directory.
        :param test_split_name: Name of test split directory under EC_Number.
        :param val_split_name: Name of validation split directory under EC_Number.
        :param batch_size: Batch size per optimization step.
        :param num_workers: Number of data‑loading workers.
        :param pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        # Store protein IDs for each split
        self.train_protein_ids: List[str] = []
        self.val_protein_ids: List[str] = []
        self.test_protein_ids: List[str] = []

        # EC mapping for this module (also copied to ProteinDataset cache in setup)
        self._ec2idx: Dict[str, int] = {}


    def _list_pids(self, split_dir: Path) -> List[str]:
        out: List[str] = []
        for d in split_dir.iterdir():
            if not d.is_dir():
                continue
            # ensure required embedding files exist
            required = [
                # d / "esmc_emb.pt",
                d / "ankh_emb_xl.pt",
                d / "prot_t5_emb.pt",
                # d / "pglm_emb.pt",
            ]
            if not all(p.exists() for p in required):
                print(f"[WARN] {d.name} missing required embedding files")
                continue
            out.append(d.name)
        return sorted(out)

    def prepare_data(self) -> None:
        """Build EC mapping and discover train/val/test protein IDs."""
        # Skip if already done
        if self.train_protein_ids and self.val_protein_ids and self.test_protein_ids:
            print(
                f"Using previously found proteins: train={len(self.train_protein_ids)}, "
                f"val={len(self.val_protein_ids)}, test={len(self.test_protein_ids)}"
            )
            return

        root = Path(self.hparams.data_dir)
        train_dir = root / "Training_set"
        val_dir = root / self.hparams.val_split_name
        test_dir = root / self.hparams.test_split_name

        if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
            raise ValueError("Missing EC_Number/Training_set, validation split directory, or test split directory")

        # EC mapping: always load from fixed pickle (no directory sweep)
        if not EC_IDX_PKL.exists():
            raise FileNotFoundError(f"EC index pickle not found at {EC_IDX_PKL}")

        # Prefer loading with pickle since the file is a plain Python dict pickle.
        # Fallback to torch.load for compatibility with alternative formats.
        try:
            with open(EC_IDX_PKL, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            obj = torch.load(EC_IDX_PKL, map_location="cpu")

        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict in {EC_IDX_PKL}, got {type(obj)}")
        self._ec2idx = dict(obj)
        print(f"Length of EC mapping: {len(self._ec2idx)}")

        # discover IDs
        self.train_protein_ids = self._list_pids(train_dir)
        self.val_protein_ids = self._list_pids(val_dir)
        self.test_protein_ids = self._list_pids(test_dir)

        if len(self.train_protein_ids) == 0:
            raise ValueError("No valid training proteins found")
        if len(self.val_protein_ids) == 0:
            raise ValueError("No valid validation proteins found")
        if len(self.test_protein_ids) == 0:
            raise ValueError("No valid test proteins found")

        print(
            f"[EC] train={len(self.train_protein_ids)}  val={len(self.val_protein_ids)}  "
            f"test={len(self.test_protein_ids)}  classes={len(self._ec2idx)}"
        )

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
            print(f"Training proteins: {len(self.train_protein_ids)}")
            print(f"Validation proteins: {len(self.val_protein_ids)}")
            print(f"Test proteins: {len(self.test_protein_ids)}")

            root = Path(self.hparams.data_dir)
            train_dir = root / "Training_set"
            val_dir = root / self.hparams.val_split_name
            test_dir = root / self.hparams.test_split_name

            self.data_train = ProteinDataset(
                data_dir=str(train_dir),
                protein_ids=self.train_protein_ids,
                split="train",
                ec2idx=self._ec2idx,
            )
            self.data_val = ProteinDataset(
                data_dir=str(val_dir),
                protein_ids=self.val_protein_ids,
                split="val",
                ec2idx=self._ec2idx,
            )
            self.data_test = ProteinDataset(
                data_dir=str(test_dir),
                protein_ids=self.test_protein_ids,
                split="test",
                ec2idx=self._ec2idx,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

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

# ======================================================================
#  Global collate function – picklable for multiprocessing 'spawn'
# ======================================================================

def protein_collate(batch):
    """Collate function for protein embeddings and labels.
    Pads per-residue embeddings to a common length and builds a boolean pad mask.
    """
    protein_ids = [it["protein_id"] for it in batch]
    sequences   = [it["sequence"]    for it in batch]
    lengths     = [it["length"]      for it in batch]   # residue counts (no CLS/EOS)

    # ----------------------------------------------------
    # 1) Pad sequence embeddings (float tensors)
    # ----------------------------------------------------
    max_len_seq = max(lengths) + 2   # CLS + residues + EOS
    esmc_emb_padded = []
    ankh_emb_padded = []
    prot_emb_padded = []
    pglm_emb_padded = []
    for it in batch:
        # Pad ESM-C embeddings
        esmc_emb = it["esmc_emb"]  # [L+2, d_esmc]
        if esmc_emb.size(0) < max_len_seq:
            esmc_emb = F.pad(esmc_emb, (0, 0, 0, max_len_seq - esmc_emb.size(0)), value=0)
        esmc_emb_padded.append(esmc_emb.float())
        
        # Pad Ankh3-XLarge embeddings (same padding logic)
        ankh_emb = it["ankh_emb"]  # [L+2, d_ankh]
        if ankh_emb.size(0) < max_len_seq:
            ankh_emb = F.pad(ankh_emb, (0, 0, 0, max_len_seq - ankh_emb.size(0)), value=0)
        ankh_emb_padded.append(ankh_emb.float())

        # Pad ProtT5 embeddings
        prot_emb = it["prot_emb"]  # [L+2, d_prot]
        if prot_emb.size(0) < max_len_seq:
            prot_emb = F.pad(prot_emb, (0, 0, 0, max_len_seq - prot_emb.size(0)), value=0)
        prot_emb_padded.append(prot_emb.float())

        # Pad PGLM embeddings
        pglm_emb = it["pglm_emb"]  # [L+2, d_pglm]
        if pglm_emb.size(0) < max_len_seq:
            pglm_emb = F.pad(pglm_emb, (0, 0, 0, max_len_seq - pglm_emb.size(0)), value=0)
        pglm_emb_padded.append(pglm_emb.float())

    # ----------------------------------------------------
    # 2) Stack embeddings + build masks
    # ----------------------------------------------------
    esmc_emb = torch.stack(esmc_emb_padded)       # [B, L_max_seq, d_esmc]
    ankh_emb = torch.stack(ankh_emb_padded)       # [B, L_max_seq, d_ankh]
    prot_emb = torch.stack(prot_emb_padded)       # [B, L_max_seq, d_prot]
    pglm_emb = torch.stack(pglm_emb_padded)       # [B, L_max_seq, d_pglm]

    labels = torch.stack([it["labels"] for it in batch])

    pad_mask = torch.tensor(
        [[i >= l + 2 for i in range(max_len_seq)] for l in lengths],
        dtype=torch.bool,
    )

    lengths_tensor = torch.tensor(lengths)

    return {
        "protein_id": protein_ids,
        "sequence": sequences,
        "esmc_emb": esmc_emb,
        "ankh_emb": ankh_emb,
        "prot_emb": prot_emb,
        "pglm_emb": pglm_emb,
        "labels": labels,
        "pad_mask": pad_mask,
        "lengths": lengths_tensor,
    }