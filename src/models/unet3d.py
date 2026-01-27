from __future__ import annotations

from typing import Any, Dict

from monai.networks.nets import UNet


def build_model(cfg: Dict[str, Any]) -> UNet:
    mcfg = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=int(mcfg["in_channels"]),
        out_channels=int(mcfg["out_channels"]),
        channels=tuple(mcfg["channels"]),
        strides=tuple(mcfg["strides"]),
        num_res_units=int(mcfg["num_res_units"]),
        norm=mcfg.get("norm", "INSTANCE"),
    )
