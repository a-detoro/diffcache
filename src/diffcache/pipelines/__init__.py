from .base import FrozenComponents, PipelineAdapter


def get_adapter(name: str) -> PipelineAdapter:
    if name == "qwen_image_edit":
        from .qwen_image_edit import QwenImageEditAdapter
        return QwenImageEditAdapter()
    if name == "flux_kontext":
        from .flux_kontext import FluxKontextAdapter
        return FluxKontextAdapter()
    if name == "omnigen2":
        from .omnigen2 import OmniGen2Adapter
        return OmniGen2Adapter()
    if name == "deepgen":
        from .deepgen import DeepGenAdapter
        return DeepGenAdapter()
    raise ValueError(f"Unknown pipeline: {name!r}")


__all__ = ["FrozenComponents", "PipelineAdapter", "get_adapter"]
