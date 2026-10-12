"""Port of the (verbatim-identical, per their own docstrings) `dataset_llvip.py`
shared by the FLUX.1-Kontext/OmniGen2/DeepGen benches in `Flux.1/`,
`OmniGen2/`, `DeepGen/` on PopOS-IA: a deterministic subset of LLVIP
RGB/thermal pairs with one fixed prompt, no train/val split or shuffle --
the point is to measure timing/VRAM, not to train a real model.

The LLVIP directory layout differs slightly between machines this project
uses (`visible_train`/`infrared_train` flat on BULL's NAS mirror vs.
`visible/train`/`infrared/train` on PopOS-IA) -- `_resolve_dirs` tries both
instead of hardcoding one, so the same `--dataset-root` convention works on
either machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image

from .base import PairedSampleDataset

_LAYOUTS = (
    ("visible_train", "infrared_train"),
    ("visible/train", "infrared/train"),
)


class LLVIPPairsDataset(PairedSampleDataset):
    def __init__(self, dataset_root: str, prompt: str, max_samples: int,
                 shard_start: int = 0, shard_count: int | None = None, ext: str = "*.jpg"):
        self.dataset_root = Path(dataset_root)
        self.prompt = prompt
        self.ext = ext
        self.visible_dir, self.infrared_dir = self._resolve_dirs()
        super().__init__(max_samples=max_samples, shard_start=shard_start, shard_count=shard_count)

    def _resolve_dirs(self) -> tuple[Path, Path]:
        for visible_rel, infrared_rel in _LAYOUTS:
            visible_dir = self.dataset_root / visible_rel
            infrared_dir = self.dataset_root / infrared_rel
            if visible_dir.is_dir() and infrared_dir.is_dir():
                return visible_dir, infrared_dir
        raise FileNotFoundError(
            f"No LLVIP visible/infrared train directories found under {self.dataset_root} "
            f"(tried {_LAYOUTS})"
        )

    def _build_sample_pairs(self) -> List[Dict]:
        visible_files = {p.stem: p for p in self.visible_dir.glob(self.ext)}
        infrared_files = {p.stem: p for p in self.infrared_dir.glob(self.ext)}
        common_stems = sorted(set(visible_files) & set(infrared_files))
        if not common_stems:
            raise FileNotFoundError(f"No common pairs between {self.visible_dir} and {self.infrared_dir}")
        return [
            {"src_path": str(visible_files[s]), "tgt_path": str(infrared_files[s]), "prompt": self.prompt, "pair_id": s}
            for s in common_stems
        ]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            "source_pil": Image.open(sample["src_path"]).convert("RGB"),
            "target_pil": Image.open(sample["tgt_path"]).convert("RGB"),
            "prompt": sample["prompt"],
        }
