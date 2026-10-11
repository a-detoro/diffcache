"""Port of `training_from_jupyter/dataset_live.py::LiveMultiViewSampleDataset`
onto the `PairedSampleDataset` ABC -- same sample-pairing logic (viewpoint
rotation prompts over `MultiViewVisibleThermalImagesHPE`), unchanged. The
original file is left untouched; this is a copy adapted to the shared
interface, not an import, so the framework has no dependency on
`training_from_jupyter/`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .base import PairedSampleDataset

logger = logging.getLogger(__name__)

CAMERA_ANGLES = {"00_17": 0, "00_16": 90, "00_15": -90}

ANGLE_PROMPTS = {
    0: {
        90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
        180: "将镜头旋转180度 Rotate the camera 180 degrees.",
        -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
    },
    90: {
        90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
        -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        -180: "将镜头旋转180度 Rotate the camera 180 degrees.",
    },
    -90: {
        90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
        -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        180: "将镜头旋转180度 Rotate the camera 180 degrees.",
    },
}


class MultiViewRotationDataset(PairedSampleDataset):
    """Qwen-Image-Edit-Angles' viewpoint-rotation task: pairs of RGB frames
    from different cameras of the same timestamp, prompted with the rotation
    needed to go from one view to the other."""

    def __init__(self, dataset_root: str, split: str = "train_val", max_samples: int = 20,
                 shard_start: int = 0, shard_count: int | None = None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        super().__init__(max_samples=max_samples, shard_start=shard_start, shard_count=shard_count)
        logger.info(
            "[multiview-dataset] %d samples loaded from %s/%s (shard [%d:%d])",
            len(self.samples), dataset_root, split, shard_start, shard_start + len(self.samples),
        )

    def _build_sample_pairs(self) -> List[Dict]:
        samples = []
        timestamp_data: Dict[str, Dict[str, Path]] = {}
        base_dir = self.dataset_root / self.split

        for camera_dir in sorted(base_dir.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            if camera_id not in CAMERA_ANGLES:
                continue
            images = sorted(camera_dir.glob("*_rgb.png"))
            for img_path in images:
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    timestamp = parts[2]
                    timestamp_data.setdefault(timestamp, {})[camera_id] = img_path

        for timestamp, cameras in timestamp_data.items():
            for src_cam, src_angle in CAMERA_ANGLES.items():
                if src_cam not in cameras:
                    continue
                src_img = cameras[src_cam]
                for tgt_cam, tgt_angle in CAMERA_ANGLES.items():
                    if src_cam == tgt_cam or tgt_cam not in cameras:
                        continue
                    tgt_img = cameras[tgt_cam]
                    angle_diff = (tgt_angle - src_angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    prompt = ANGLE_PROMPTS.get(src_angle, {}).get(
                        angle_diff, f"将镜头旋转{angle_diff}度 Rotate the camera {angle_diff} degrees."
                    )
                    samples.append({
                        "src_path": str(src_img),
                        "tgt_path": str(tgt_img),
                        "prompt": prompt,
                        "timestamp": timestamp,
                    })
        return samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            "source_pil": Image.open(sample["src_path"]).convert("RGB"),
            "target_pil": Image.open(sample["tgt_path"]).convert("RGB"),
            "prompt": sample["prompt"],
        }
