import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: tuple[int, int] | None = None,
    ):
        super().__init__()
        modules = [
            nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        ]
        if pool:
            modules.append(nn.MaxPool2d(kernel_size=pool, stride=pool))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x)


class CNN_v2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvLayer(1, 32, pool=(2, 2)),
            ConvLayer(32, 64, pool=(2, 2)),
            ConvLayer(64, 128),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
