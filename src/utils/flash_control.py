import torch
import warnings

def maybe_enable_flash_attention(force: bool = False) -> None:
    """Enable Flash-/efficient-SDPA back-ends when they are available.

    Parameters
    ----------
    force : bool, default False
        If ``True`` raise ``RuntimeError`` when FlashAttention kernels are not
        present or cannot be enabled.  If ``False`` we just issue a warning and
        fall back to the standard math implementation.
    """
    # CPU or MPS execution – nothing to do.
    if not torch.cuda.is_available():
        print("Running without Flash Attention because CUDA is not available")
        return

    # Check whether the wheel was compiled with the Triton flash kernels.
    have_flash = torch.backends.cuda.is_flash_attention_available()

    if not have_flash:
        if force:
            raise RuntimeError(
                "FlashAttention kernels are not available in this PyTorch build."
            )
        warnings.warn(
            "Running without FlashAttention because kernels are missing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    print("Running with Flash Attention")
    torch.backends.cuda.enable_flash_sdp(True)          # Fastest (Ampere+)
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Triton kernels
    torch.backends.cuda.enable_math_sdp(True)           # Always available 