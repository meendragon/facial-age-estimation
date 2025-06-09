import torch
import torch.nn as nn
import math
from torch.nn.modules.utils import _triple

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel = (1, kernel_size[1], kernel_size[2])
        temporal_kernel = (kernel_size[0], 1, 1)

        spatial_stride = (1, stride[1], stride[2])
        temporal_stride = (stride[0], 1, 1)

        spatial_padding = (0, padding[1], padding[2])
        temporal_padding = (padding[0], 0, 0)

        intermed_channels = int(math.floor(
            (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)
        ))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class MiniR2Plus1D(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super(MiniR2Plus1D, self).__init__()

        self.block1 = SpatioTemporalConv(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.block2 = SpatioTemporalConv(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.block3 = SpatioTemporalConv(128, feature_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
    def forward(self, x):  # (B, C=3, T, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
