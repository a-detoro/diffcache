"""Hardware quirk detection -- generalizes the Volta workarounds that used to be
three flags (`--disable_cudnn --dtype fp16 --dit_compute_dtype fp32`) every
caller had to remember to pass by hand on BULL (see
feedback_bull_volta_quirks.md). Declarative: look at the GPU's compute
capability once, apply what that generation needs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareQuirks:
    disable_cudnn: bool = False
    force_storage_dtype: str | None = None  # e.g. "fp16" on Volta, which has no bf16 tensor cores
    force_nf4_compute_dtype: str | None = None  # e.g. "fp32" on Volta, see below


# Compute capability (major, minor) -> quirks. Volta (7.0) is the only
# generation needing workarounds among the hardware used for the JSC paper
# (Ampere/Blackwell are bf16-native and have a working cuDNN engine for the
# Conv3d ops hit here). Extend this table as new hardware is validated --
# don't guess for an untested generation, leave it with no quirks and let it
# fail loudly if something's actually wrong, rather than silently "fixing" a
# problem that may not exist there.
_VOLTA = (7, 0)

_QUIRKS_BY_CAPABILITY: dict[tuple[int, int], HardwareQuirks] = {
    _VOLTA: HardwareQuirks(
        disable_cudnn=True,
        force_storage_dtype="fp16",
        force_nf4_compute_dtype="fp32",
    ),
}


def detect_compute_capability(device: str | torch.device = "cuda:0") -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def detect_quirks(device: str | torch.device = "cuda:0") -> HardwareQuirks:
    capability = detect_compute_capability(device)
    quirks = _QUIRKS_BY_CAPABILITY.get(capability, HardwareQuirks())
    if quirks != HardwareQuirks():
        logger.warning(
            "Hardware quirks active for compute capability %s (%s): %s",
            capability, torch.cuda.get_device_name(device), quirks,
        )
    return quirks


def apply_quirks(precision, device: str | torch.device = "cuda:0"):
    """Mutates a `config.PrecisionConfig` in place to match this hardware's
    quirks, and flips the global `torch.backends.cudnn.enabled` flag.
    Returns the quirks found, for logging/reporting.

    Only overrides fields the caller left at their default -- an explicit
    `--dtype fp16` or `--compute-dtype fp32` the user already passed is never
    silently clobbered, since on non-Volta hardware those same flags are
    valid, deliberate choices (e.g. testing fp16 on Ampere for comparison).
    """
    quirks = detect_quirks(device)

    if quirks.disable_cudnn:
        torch.backends.cudnn.enabled = False
        logger.warning("cuDNN disabled (torch.backends.cudnn.enabled = False)")

    if quirks.force_storage_dtype and precision.storage_dtype == "bf16":
        logger.warning(
            "Forcing storage dtype bf16 -> %s (this GPU generation has no bf16 tensor cores)",
            quirks.force_storage_dtype,
        )
        precision.storage_dtype = quirks.force_storage_dtype

    if (
        quirks.force_nf4_compute_dtype
        and precision.quant_type == "nf4"
        and precision.compute_dtype is None
    ):
        logger.warning(
            "Forcing NF4 compute dtype -> %s (bitsandbytes' Linear4bit downcasts its "
            "output to the INPUT tensor's dtype regardless of bnb_4bit_compute_dtype -- "
            "this model's activations overflow fp16 on this hardware, see "
            "feedback_bull_volta_quirks.md)",
            quirks.force_nf4_compute_dtype,
        )
        precision.compute_dtype = quirks.force_nf4_compute_dtype

    return quirks
