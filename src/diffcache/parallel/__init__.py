from .base import ParallelBackend, ParallelEngine


def get_backend(name: str) -> ParallelBackend:
    if name == "zero2":
        from .zero import ZeroBackend
        return ZeroBackend()
    if name == "ddp":
        from .ddp import DDPBackend
        return DDPBackend()
    raise ValueError(f"Unknown parallel backend: {name!r} (use 'zero2' or 'ddp')")


__all__ = ["ParallelBackend", "ParallelEngine", "get_backend"]
