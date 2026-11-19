"""CNN 2D sobre a representação espectro-temporal (features x frames).

Trata a matriz de LFCC/log-Mel como uma "imagem" de 1 canal. As convoluções
capturam padrões locais no plano tempo-frequência (artefatos de vocoder,
descontinuidades espectrais), e o pooling global agrega ao longo da locução.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(self, n_features: int, n_frames: int,
                 channels=(32, 64, 128, 128), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T) -> (B, 1, F, T)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)  # (B,)
