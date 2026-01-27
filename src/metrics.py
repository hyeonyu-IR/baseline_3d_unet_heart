from __future__ import annotations

import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def dice_metric():
    # We care about left atrium (label=1), not background.
    return DiceMetric(include_background=False, reduction="mean")


def post_pred():
    return AsDiscrete(argmax=True, to_onehot=2)


def post_label():
    return AsDiscrete(to_onehot=2)
