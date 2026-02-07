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


class CRNN(nn.Module):
    def __init__(
        self,
        height: int,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        # In: B, 3, H, W
        self.cnn = nn.Sequential(
            ConvLayer(3, 32, pool=(2, 2)),  # B, 32, H/2, W/2
            ConvLayer(32, 64, pool=(2, 2)),  # B, 64, H/4, W/4
            ConvLayer(64, 128, pool=(2, 1)),  # B, 128, H/8, W/4
            ConvLayer(128, 256, pool=(2, 1)),  # B, 128, H/16, W/4
        )
        # Out: B, 128, H/16, W/4

        # In: B, W/4, 128 * H/16
        self.rnn = nn.LSTM(
            input_size=256 * (height // 16),
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        # Out: B, W/4, d_h * 2

        # In: B, W/4, d_h * 2
        self.transcription = nn.Linear(rnn_hidden * 2, num_classes)
        # Out: B, W/4, d_c

    def forward(self, x):
        x = self.cnn(x)

        B, C, H, W = x.shape

        x = x.permute(0, 3, 1, 2)  # B, W/8, 128, H/4
        x = x.reshape(B, W, -1)  # B, W/8, 128 * H/4

        x, _ = self.rnn(x)  # B, W/8, d_h * 2

        x = self.transcription(x)  # B, W/8, d_c
        return x
