from collections import OrderedDict

import torch
import torch.nn as nn


def _norm_layer(num_features, affine=True, track_running_stats=True, norm='bn'):
    if norm == 'bn':
        return nn.BatchNorm2d(
            num_features=num_features,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    if norm == 'in':
        return nn.InstanceNorm2d(
            num_features=num_features,
            affine=False,
            track_running_stats=False,
        )
    raise ValueError('Unsupported normalization: {}'.format(norm))


def _conv_block(in_channels, features, name, affine=True, track_running_stats=True, norm='bn'):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "_conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn1", _norm_layer(features, affine, track_running_stats, norm)),
                (name + "_relu1", nn.ReLU(inplace=True)),
                (
                    name + "_conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn2", _norm_layer(features, affine, track_running_stats, norm)),
                (name + "_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, features, affine=True, track_running_stats=True, norm='bn'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)
        self.bn1 = _norm_layer(features, affine, track_running_stats, norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm_layer(features, affine, track_running_stats, norm)
        if in_channels == features:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=1, bias=False),
                _norm_layer(features, affine, track_running_stats, norm),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SCSEBlock, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels,
                 affine=True, track_running_stats=True, norm='bn'):
        super(AttentionGate, self).__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            _norm_layer(inter_channels, affine, track_running_stats, norm),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _norm_layer(inter_channels, affine, track_running_stats, norm),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * attention


class ResAttUNet(nn.Module):
    """U-Net variant for FedCLAM backbone ablations.

    It keeps the original decoder by default, replaces encoder blocks with
    residual blocks, and optionally enables scSE, skip attention gates, and
    nnU-Net style deep supervision heads.
    """

    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None, norm='bn',
                 attention=None, deep_supervision=False):
        super(ResAttUNet, self).__init__()
        del aug_method

        valid_attention = [None, 'none', 'scse', 'gate', 'scse_gate']
        if attention not in valid_attention:
            raise ValueError('attention must be one of {}'.format(valid_attention))

        self.attention = None if attention == 'none' else attention
        self.deep_supervision = deep_supervision
        features = init_features

        self.encoder1 = ResidualBlock(in_channels, features, affine, track_running_stats, norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features, features * 2, affine, track_running_stats, norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features * 2, features * 4, affine, track_running_stats, norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features * 4, features * 8, affine, track_running_stats, norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(features * 8, features * 16, affine, track_running_stats, norm)

        if self.attention in ['scse', 'scse_gate']:
            self.scse3 = SCSEBlock(features * 4)
            self.scse4 = SCSEBlock(features * 8)
            self.scse_bottleneck = SCSEBlock(features * 16)
        else:
            self.scse3 = nn.Identity()
            self.scse4 = nn.Identity()
            self.scse_bottleneck = nn.Identity()

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = _conv_block((features * 8) * 2, features * 8, name="dec4",
                                    affine=affine, track_running_stats=track_running_stats, norm=norm)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = _conv_block((features * 4) * 2, features * 4, name="dec3",
                                    affine=affine, track_running_stats=track_running_stats, norm=norm)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = _conv_block((features * 2) * 2, features * 2, name="dec2",
                                    affine=affine, track_running_stats=track_running_stats, norm=norm)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = _conv_block(features * 2, features, name="dec1",
                                    affine=affine, track_running_stats=track_running_stats, norm=norm)

        if self.attention in ['gate', 'scse_gate']:
            self.gate4 = AttentionGate(features * 8, features * 8, features * 4,
                                       affine, track_running_stats, norm)
            self.gate3 = AttentionGate(features * 4, features * 4, features * 2,
                                       affine, track_running_stats, norm)
            self.gate2 = AttentionGate(features * 2, features * 2, features,
                                       affine, track_running_stats, norm)
            self.gate1 = AttentionGate(features, features, max(features // 2, 1),
                                       affine, track_running_stats, norm)
        else:
            self.gate4 = self.gate3 = self.gate2 = self.gate1 = None

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        if self.deep_supervision:
            self.ds2 = nn.Conv2d(in_channels=features * 2, out_channels=out_channels, kernel_size=1)
            self.ds3 = nn.Conv2d(in_channels=features * 4, out_channels=out_channels, kernel_size=1)

    def _skip(self, gate, skip, gate_module):
        if gate_module is None:
            return skip
        return gate_module(gate, skip)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.scse3(self.encoder3(self.pool2(enc2)))
        enc4 = self.scse4(self.encoder4(self.pool3(enc3)))
        bottleneck = self.scse_bottleneck(self.bottleneck(self.pool4(enc4)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self._skip(dec4, enc4, self.gate4)), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self._skip(dec3, enc3, self.gate3)), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self._skip(dec2, enc2, self.gate2)), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self._skip(dec1, enc1, self.gate1)), dim=1)
        dec1 = self.decoder1(dec1)
        main = self.conv(dec1)

        if not self.deep_supervision:
            return main
        return main, self.ds2(dec2), self.ds3(dec3)
