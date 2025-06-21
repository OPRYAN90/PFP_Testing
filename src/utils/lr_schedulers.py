import math
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Cosine decay with linear warm-up.

    This is a minimal extraction of the implementation that ships with
    HuggingFace *transformers* so that we do **not** need to depend on that
    package (or on *torchtune*) just for one utility function.

    Args:
        optimizer: Wrapped optimizer whose LR we schedule.
        num_warmup_steps: How many steps linearly increase LR from 0 → base LR.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 → single half-cosine).
        last_epoch: Index of last epoch when resuming training.

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` instance.
    """

    def lr_lambda(current_step: int):
        # Linear warm-up
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        )
        return max(0.0, cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch) 