from torch.optim.swa_utils import get_ema_multi_avg_fn 
from .weight_averaging import WeightAveraging

#NOTE: NEVER RESUME A CHECKPOINT FOR TRAINING IF WARMUP PERIOD HAS NOT BEEN COMPLETED

class EMAWeightAveraging(WeightAveraging):
    def __init__(
        self,
        decay: float = 0.999,
        start_step: int = 0,
        device=None,
        use_buffers: bool = True,
    ):
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
        )
        self.start_step = int(start_step)
        self._ema_bootstrapped = False
        print(
            f"EMAWeightAveraging initialized with decay={decay}, start_step={start_step}, device={device}, use_buffers={use_buffers}"
        )

    # Update once per optimizer step, after warmup
    def should_update(self, step_idx=None, epoch_idx=None) -> bool:
        return (step_idx is not None) and (step_idx >= self.start_step)

    # Bootstrap EMA at first eligible update (copy current -> avg, no mixing with init)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(step_idx=step_idx):
            if not self._ema_bootstrapped:
                self._average_model.module.load_state_dict(pl_module.state_dict(), strict=False)
                self._ema_bootstrapped = True
                print(f"IMPORTANT: Bootstrapping EMA, switched self._ema_bootstrapped={self._ema_bootstrapped}")
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    # Only use EMA weights for eval once bootstrapped
    def on_validation_epoch_start(self, trainer, pl_module):
        if self._average_model is not None and self._ema_bootstrapped:
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._average_model is not None and self._ema_bootstrapped:
            self._swap_models(pl_module)

    def on_train_end(self, trainer, pl_module):
        if self._average_model is not None and self._ema_bootstrapped:
            self._copy_average_to_current(pl_module)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict):
        # Restore averaged model + WA state
        super().on_load_checkpoint(trainer, pl_module, checkpoint)
        self._ema_bootstrapped = True
        print(f"TESTING: EMA bootstrapped, testing time: self._ema_bootstrapped={self._ema_bootstrapped}")