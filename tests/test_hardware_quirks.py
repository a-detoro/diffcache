import torch

from cachezero import hardware
from cachezero.config import PrecisionConfig


def test_volta_quirks_force_fp16_and_fp32_nf4_compute(monkeypatch):
    monkeypatch.setattr(hardware, "detect_compute_capability", lambda device: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla V100-PCIE-32GB (mocked)")
    original_cudnn = torch.backends.cudnn.enabled
    try:
        precision = PrecisionConfig(quant_type="nf4", storage_dtype="bf16")
        quirks = hardware.apply_quirks(precision, device="cuda:0")

        assert quirks.disable_cudnn is True
        assert torch.backends.cudnn.enabled is False
        assert precision.storage_dtype == "fp16"
        assert precision.compute_dtype == "fp32"
    finally:
        torch.backends.cudnn.enabled = original_cudnn


def test_non_volta_hardware_leaves_precision_untouched(monkeypatch):
    monkeypatch.setattr(hardware, "detect_compute_capability", lambda device: (8, 6))  # Ampere
    precision = PrecisionConfig(quant_type="nf4", storage_dtype="bf16")
    quirks = hardware.apply_quirks(precision, device="cuda:0")

    assert quirks.disable_cudnn is False
    assert precision.storage_dtype == "bf16"
    assert precision.compute_dtype is None


def test_explicit_compute_dtype_is_never_overridden_by_quirks(monkeypatch):
    monkeypatch.setattr(hardware, "detect_compute_capability", lambda device: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla V100-PCIE-32GB (mocked)")
    original_cudnn = torch.backends.cudnn.enabled
    try:
        precision = PrecisionConfig(quant_type="nf4", storage_dtype="fp16", compute_dtype="bf16")
        hardware.apply_quirks(precision, device="cuda:0")
        # The caller explicitly chose bf16 compute (e.g. to test it on Volta on purpose) --
        # the quirk table must not silently clobber a deliberate choice.
        assert precision.compute_dtype == "bf16"
    finally:
        torch.backends.cudnn.enabled = original_cudnn
