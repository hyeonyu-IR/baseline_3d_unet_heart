from __future__ import annotations

from typing import Any, Dict

import torch
from monai.inferers import sliding_window_inference


@torch.no_grad()
def infer_logits(model: torch.nn.Module, x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    icfg = cfg["infer"]
    roi_size = tuple(icfg["roi_size"])
    sw_batch_size = int(icfg["sw_batch_size"])
    overlap = float(icfg["overlap"])
    return sliding_window_inference(x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap)
