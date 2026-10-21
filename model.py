from dataclasses import dataclass

import torch

RETINA_SIZE = 36

@dataclass
class ModelResult:
    x: int
    y: int
    scale: float
    probability: float # that it is a face


@dataclass
class ImageLayer:
    image: torch.Tensor
    scale: float
    width: int
    height: int


@dataclass
class BorderBox:
    x: int
    y: int
    certainity: float
    size: int