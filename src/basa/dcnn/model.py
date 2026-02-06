import torch
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


class SplitHorizontal(nn.Module):
    def __init__(self, num_letters: int):
        super().__init__()
        self.num_letters = num_letters

    def forward(self, x):
        # torch.chunk(x, NUM_LETTERS, dim=3)
        B, C, H, W = x.shape
        split_w = W // self.num_letters
        x = x.reshape(B, C, H, self.num_letters, split_w)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(B, self.num_letters, -1)
        return x


class CNN_v3(nn.Module):
    def __init__(self, num_letters: int, len_letters: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvLayer(3, 32, pool=(2, 2)),
            ConvLayer(32, 64, pool=(2, 2)),
            ConvLayer(64, 128),
            SplitHorizontal(num_letters),
            nn.Linear(128 * 3 * 15, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Linear(512, len_letters),
                )
                for _ in range(num_letters)
            ]
        )

    def forward(self, x):
        features = self.features(x)
        B, S, F = features.shape  # Batch, Split, Features

        return torch.stack(
            [
                classifier(features[:, i, :].reshape(B, F))
                for (i, classifier) in enumerate(self.classifiers)
            ],
            dim=1,
        )
