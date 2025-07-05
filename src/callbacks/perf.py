# src/callbacks/perf.py
import time
import json
from pathlib import Path
from lightning.pytorch.callbacks import Callback

class BatchSizeFn:
    """Callable class to extract batch size from protein batch dictionary."""
    
    def __call__(self, batch):
        """Extract batch size from protein batch dictionary."""
        if isinstance(batch, dict):
            # For protein batches, use the length of protein_id list or any tensor's first dimension
            if "protein_id" in batch:
                return len(batch["protein_id"])
            elif "sequence_emb" in batch:
                return batch["sequence_emb"].size(0)
            elif "labels" in batch:
                return batch["labels"].size(0)
        return 1  # fallback

def get_batch_size(batch):
    """Extract batch size from protein batch dictionary."""
    if isinstance(batch, dict):
        # For protein batches, use the length of protein_id list or any tensor's first dimension
        if "protein_id" in batch:
            return len(batch["protein_id"])
        elif "sequence_emb" in batch:
            return batch["sequence_emb"].size(0)
        elif "labels" in batch:
            return batch["labels"].size(0)
    return 1  # fallback

class BatchTimer(Callback):
    """Callback that
    1. Logs dataloader-waiting time and forward-pass time to the logger (as before)
    2. Stores both timings in memory **and** persists them to a JSON file *after every update*.

    This way, even if training is interrupted you still have a full history of timings on disk.
    Only the global-zero rank will write to disk to avoid race conditions in DDP runs.
    """

    def __init__(self, json_path: str = "batch_times.json", flush_every_n: int = 1):
        """
        Parameters
        ----------
        json_path: str
            Path to the JSON file that will hold two keys: ``dataloader_times`` and ``forward_times``.
            The file is (re)written after every ``flush_every_n`` updates.
        flush_every_n: int
            Frequency (in number of recorded values) at which the JSON file is rewritten.
        """
        super().__init__()
        self.json_path = Path(json_path)
        self.flush_every_n = flush_every_n

        # In-memory storage
        self.dataloader_times = []
        self.forward_times = []
        self.prep_times = []  # complete batch-prep (dataset + collate)
        # New: time spent on the MSA embedding step for each batch
        self.msa_times = []
        # New: detailed timing for A3M parsing and MSA computation
        self.a3m_parse_times = []
        self.msa_compute_times = []

        # Internal timers (initialize as floats to avoid Optional complaints)
        self.prev_batch_end: float = 0.0
        self._t0: float = 0.0

    # ---------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------
    def _maybe_flush(self, trainer):
        """Write current lists to disk if required (rank-0 only)."""
        if (trainer.is_global_zero  # only one process writes
                and len(self.forward_times) % self.flush_every_n == 0):
            tmp = {
                "dataloader_times": self.dataloader_times,
                "forward_times": self.forward_times,
                "prep_times": self.prep_times,
                "msa_times": self.msa_times,
                "a3m_parse_times": self.a3m_parse_times,
                "msa_compute_times": self.msa_compute_times,
            }
            # We overwrite the whole file each time to keep a valid JSON at all times.
            self.json_path.write_text(json.dumps(tmp, indent=2))

    # ---------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------
    def on_train_start(self, trainer, *_):
        # Restore previous state if file exists (useful for resumed runs)
        if self.json_path.exists():
            try:
                data = json.loads(self.json_path.read_text())
                self.dataloader_times = data.get("dataloader_times", [])
                self.forward_times = data.get("forward_times", [])
                self.prep_times = data.get("prep_times", [])
                self.msa_times = data.get("msa_times", [])
                self.a3m_parse_times = data.get("a3m_parse_times", [])
                self.msa_compute_times = data.get("msa_compute_times", [])
            except Exception:
                # Any corruption → start fresh (don't crash training)
                self.dataloader_times = []
                self.forward_times = []
                self.prep_times = []
                self.msa_times = []
                self.a3m_parse_times = []
                self.msa_compute_times = []

        self.prev_batch_end = time.perf_counter()  # t = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        dl_time = now - self.prev_batch_end  # time waiting for next batch

        # Log via logger
        if trainer.logger:
            trainer.logger.log_metrics({"time/dataloader": dl_time}, step=trainer.global_step)

        # Persist
        self.dataloader_times.append(dl_time)
        self._maybe_flush(trainer)

        # Extract prep time (if available) and log it
        prep_time = batch.get("prep_time")
        if prep_time is not None:
            self.prep_times.append(prep_time)
            if trainer.logger:
                trainer.logger.log_metrics({"time/prep": prep_time}, step=trainer.global_step)
            self._maybe_flush(trainer)

        # Extract MSA embedding time (if available)
        msa_time = batch.get("msa_time")
        if msa_time is not None:
            self.msa_times.append(msa_time)
            if trainer.logger:
                trainer.logger.log_metrics({"time/msa": msa_time}, step=trainer.global_step)
            self._maybe_flush(trainer)

        # Extract A3M parsing time (if available)
        a3m_parse_time = batch.get("a3m_parse_time")
        if a3m_parse_time is not None:
            self.a3m_parse_times.append(a3m_parse_time)
            if trainer.logger:
                trainer.logger.log_metrics({"time/a3m_parse": a3m_parse_time}, step=trainer.global_step)
            self._maybe_flush(trainer)

        # Extract MSA compute time (if available)
        msa_compute_time = batch.get("msa_compute_time")
        if msa_compute_time is not None:
            self.msa_compute_times.append(msa_compute_time) 
            if trainer.logger:
                trainer.logger.log_metrics({"time/msa_compute": msa_compute_time}, step=trainer.global_step)
            self._maybe_flush(trainer)

        # Start forward timer
        self._t0 = now

    def on_before_backward(self, trainer, *_):
        fwd_time = time.perf_counter() - self._t0  # forward(+loss) done
        if trainer.logger:
            trainer.logger.log_metrics({"time/forward": fwd_time}, step=trainer.global_step)

        self.forward_times.append(fwd_time)
        self._maybe_flush(trainer)

        # ------------------------------------------------------------------
        # Log MSA compute time that was measured **during** the forward pass.
        # The forward pass stores it as an attribute on the LightningModule so
        # we can safely access it here *after* the computation is finished.
        # ------------------------------------------------------------------
        pl_module = trainer.lightning_module
        msa_compute_time = getattr(pl_module, "_last_msa_compute_time", None)
        if msa_compute_time is not None:
            self.msa_compute_times.append(msa_compute_time)
            if trainer.logger:
                trainer.logger.log_metrics({"time/msa_compute": msa_compute_time}, step=trainer.global_step)
            # Flush periodically
            self._maybe_flush(trainer)
            # Clear to avoid double-logging when using gradient accumulation
            pl_module._last_msa_compute_time = None

    def on_train_batch_end(self, trainer, *_):
        # Mark end-of-step so next dataloader timing is correct
        self.prev_batch_end = time.perf_counter() 