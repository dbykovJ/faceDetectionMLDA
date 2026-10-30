import torch
from torch import nn


def conv_bn(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """Two 3x3 convs with a skip connection (same channels)."""
    def __init__(self, ch):
        super().__init__()
        self.c1 = conv_bn(ch, ch)
        self.c2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                                nn.BatchNorm2d(ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.c2(self.c1(x)))


class Net(nn.Module):
    """
    Fully convolutional face detector.
    - Trained on 36x36 crops -> output is 1x1 (one logit, optional 4 box offsets).
    - Run on a whole pyramid level -> output is a score map, one score per
      36x36 window, with a stride of 4 pixels.
    Pad each pyramid level so H and W are multiples of 4 for exact alignment.
    """
    def __init__(self, box_regression=False):
        super().__init__()
        # Stage 1: full resolution (36x36)
        self.s1 = nn.Sequential(conv_bn(1, 16), conv_bn(16, 16))
        # Stage 2: 18x18
        self.s2 = nn.Sequential(conv_bn(16, 32), ResBlock(32))
        # Stage 3: 9x9
        self.s3 = nn.Sequential(conv_bn(32, 64), ResBlock(64))

        self.pool = nn.MaxPool2d(2, 2)
        # Fixed (non-adaptive) pooling so every level lands on the 9x9 grid
        # and the network stays translation-equivariant.
        self.lat1 = nn.Sequential(nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True), nn.AvgPool2d(4, 4))
        self.lat2 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True), nn.AvgPool2d(2, 2))

        fused = 64 + 16 + 16
        self.reduce = conv_bn(fused, 48, k=1, p=0)

        # "Fully connected" layer written as a 9x9 conv: on a 36x36 input it
        # collapses to 1x1 (identical to flatten + Linear); on a bigger image
        # it slides over every window.
        self.window = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=9, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.cls = nn.Conv2d(64, 1, 1)                       # face logit
        self.box = nn.Conv2d(64, 4, 1) if box_regression else None

    def forward(self, x):
        x1 = self.s1(x)                    # 36x36
        x2 = self.s2(self.pool(x1))        # 18x18
        x3 = self.s3(self.pool(x2))        # 9x9
        f = torch.cat([self.lat1(x1), self.lat2(x2), x3], dim=1)
        h = self.window(self.reduce(f))
        logit = self.cls(h)
        if self.box is not None:
            return logit, self.box(h)
        return logit                        # (N,1,h,w); use BCEWithLogits / focal loss

