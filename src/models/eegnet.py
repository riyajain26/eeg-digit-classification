"""
EEGNet (Lawhern et al., 2018) - compact CNN for EEG classification.
From Notebook 06, Section 4. Fully parameterized: n_channels/n_samples are
required (determined by your data), everything else has defaults matching
what worked for Stage 1 but can be overridden per experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, dropout: float = 0.5):
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2

        # Block 1: temporal convolution (learns frequency-selective filters)
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length),
                                        padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution (learns spatial filters per temporal filter)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Separable convolution (combines features efficiently)
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.pointwise_conv = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_size = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.bn1(self.temporal_conv(x))
        x = self.bn2(self.depthwise_conv(x))
        x = F.elu(x)
        x = self.dropout1(self.pool1(x))
        x = self.pointwise_conv(self.separable_conv(x))
        x = F.elu(self.bn3(x))
        x = self.dropout2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
