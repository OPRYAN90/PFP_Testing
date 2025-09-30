import time
import json
from pathlib import Path
from lightning.pytorch.callbacks import Callback

class BatchTimer(Callback):
    """Callback that logs dataloader-waiting time and forward-pass time to the logger."""
    def __init__(self, json_path: str = "batch_times.json", flush_every_n: int = 1):
        super().__init__()
        self.json_path = Path(json_path)
        self.flush_every_n = flush_every_n
        self.dataloader_times = []
        self.forward_times = []
        self.prev_batch_end: float = 0.0
        self._t0: float = 0.0

    def _maybe_flush(self, trainer):
        if (trainer.is_global_zero
                and len(self.forward_times) % self.flush_every_n == 0):
            tmp = {
                "dataloader_times": self.dataloader_times,
                "forward_times": self.forward_times,
            }
            self.json_path.write_text(json.dumps(tmp, indent=2))

    def on_train_start(self, trainer, *_):
        if self.json_path.exists():
            try:
                data = json.loads(self.json_path.read_text())
                self.dataloader_times = data.get("dataloader_times", [])
                self.forward_times = data.get("forward_times", [])
            except Exception:
                self.dataloader_times = []
                self.forward_times = []
        self.prev_batch_end = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        dl_time = now - self.prev_batch_end
        if trainer.logger:
            trainer.logger.log_metrics({"time/dataloader": dl_time}, step=trainer.global_step)
        self.dataloader_times.append(dl_time)
        self._maybe_flush(trainer)
        self._t0 = now

    def on_before_backward(self, trainer, *_):
        fwd_time = time.perf_counter() - self._t0
        if trainer.logger:
            trainer.logger.log_metrics({"time/forward": fwd_time}, step=trainer.global_step)
        self.forward_times.append(fwd_time)
        self._maybe_flush(trainer)

    def on_train_batch_end(self, trainer, *_):
        self.prev_batch_end = time.perf_counter() 