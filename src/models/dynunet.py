# src/models/dynunet.py
from __future__ import annotations

from typing import Any, Dict, Sequence

from monai.networks.nets import DynUNet


def build_dynunet(cfg: Dict[str, Any]) -> DynUNet:
    """
    nnU-Net–style dynamic U-Net (MONAI DynUNet).

    Notes
    - We start with deep_supervision=False so that the network returns a single tensor
      of shape (B, C, D, H, W), compatible with your existing training pipeline.
    - You can later enable deep supervision once the baseline DynUNet benchmark is stable.
    """
    mcfg = cfg["model"]

    in_channels = int(mcfg["in_channels"])
    out_channels = int(mcfg["out_channels"])

    # These must match the number of downsampling stages:
    # len(strides) == len(kernels) == number of stages
    strides: Sequence[int] = tuple(mcfg.get("strides", [2, 2, 2, 2]))
    kernels: Sequence[int] = tuple(mcfg.get("kernels", [3, 3, 3, 3, 3]))

    # DynUNet expects kernel_size per stage including bottleneck (hence often len(kernels)=len(strides)+1)
    # A common pattern:
    # strides: [2,2,2,2] -> 4 downsamples
    # kernels: [3,3,3,3,3] -> 5 stages (incl bottleneck)
    if len(kernels) != len(strides):
        raise ValueError(
            f"DynUNet expects len(kernels) == len(strides), got kernels={len(kernels)}, strides={len(strides)}"
    )


    # Filter sizes: one per stage (incl bottleneck). Common nnU-Net-like progression.
    # If you want config control, add 'filters' in YAML.
    filters = tuple(mcfg.get("filters", [32, 64, 128, 256, 320]))
    if len(filters) != len(kernels):
        raise ValueError(
            f"len(filters) must equal len(kernels). Got filters={len(filters)}, kernels={len(kernels)}"
        )

    # By default we keep deep supervision off
    deep_supervision = bool(mcfg.get("deep_supervision", False))

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides,
        filters=filters,
        norm_name=mcfg.get("norm", "INSTANCE"),
        act_name=mcfg.get("act", ("leakyrelu", {"negative_slope": 0.01, "inplace": True})),
        deep_supervision=deep_supervision,
        deep_supr_num=int(mcfg.get("deep_supr_num", 1)),
        res_block=bool(mcfg.get("res_block", True)),
    )
    return net
