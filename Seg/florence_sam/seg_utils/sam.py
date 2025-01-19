import os
from typing import Any
from pathlib import Path
import numpy as np
import supervision as sv
import torch
import hydra
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# SAM_CHECKPOINT = os.path.realpath(os.path.join(__file__, '../../checkpoints/sam2_hiera_small.pt'))
SAM_CHECKPOINT = str(Path(__file__).parent.parent.joinpath("checkpoints/sam2_hiera_small.pt"))
# print(f'SAM_CHECKPOINT: {SAM_CHECKPOINT}')
SAM_CONFIG = "sam2_hiera_s.yaml"


def load_sam_image_model(
        device: torch.device,
        config: str = SAM_CONFIG,
        checkpoint: str = SAM_CHECKPOINT
) -> SAM2ImagePredictor:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    path_dir = Path(__file__).parent.parent.joinpath("configs")
    hydra.initialize_config_dir(config_dir=str(path_dir))
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(sam_model=model)


def load_sam_video_model(
        device: torch.device,
        config: str = SAM_CONFIG,
        checkpoint: str = SAM_CHECKPOINT
) -> Any:
    return build_sam2_video_predictor(config, checkpoint, device=device)


def run_sam_inference(
        model: Any,
        image: Image,
        detections: sv.Detections
) -> sv.Detections:
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    mask, score, _ = model.predict(box=detections.xyxy, multimask_output=False)

    # dirty fix; remove this later
    if len(mask.shape) == 4:
        mask = np.squeeze(mask)

    detections.mask = mask.astype(bool)
    return detections
