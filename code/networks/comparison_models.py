import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_complex(x):
    return x[:, 0] + 1j * x[:, 1]


def _to_channels(x):
    return torch.stack([x.real, x.imag], dim=1)


def _coil_data_consistency(image, zero_fill, mask, coil_map):
    image_complex = _to_complex(image)
    zero_fill_complex = _to_complex(zero_fill)

    image_coils = image_complex.unsqueeze(1) * coil_map
    zero_fill_coils = zero_fill_complex.unsqueeze(1) * coil_map

    pred_kspace = torch.fft.fft2(image_coils, norm="ortho")
    ref_kspace = torch.fft.fft2(zero_fill_coils, norm="ortho")

    mask = mask.to(dtype=torch.bool)
    while mask.dim() < pred_kspace.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred_kspace)

    merged_kspace = torch.where(mask, ref_kspace, pred_kspace)
    merged_image = torch.fft.ifft2(merged_kspace, norm="ortho")
    merged_image = torch.sum(merged_image * torch.conj(coil_map), dim=1)
    return _to_channels(merged_image).type_as(image)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class MultiScalePriorNet(nn.Module):
    def __init__(self, channels=2, base_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.GELU(),
            ResidualConvBlock(base_channels),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(base_channels * 4),
            ResidualConvBlock(base_channels * 4),
        )
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, channels, 3, padding=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.bottleneck(x2)
        x = self.up1(x, x1)
        x = self.up2(x, x0)
        return self.head(x)


class FrequencyBlock(nn.Module):
    def __init__(self, channels=2, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels * 2, 1),
        )

    def forward(self, x):
        freq = torch.fft.fft2(x, norm="ortho")
        freq_feat = torch.cat([freq.real, freq.imag], dim=1)
        delta = self.net(freq_feat)
        real_delta, imag_delta = torch.chunk(delta, 2, dim=1)
        freq = freq + torch.complex(real_delta, imag_delta)
        return torch.fft.ifft2(freq, norm="ortho").real


class MDPGCascade(nn.Module):
    def __init__(self, channels=2, base_channels=32):
        super().__init__()
        self.image_prior = MultiScalePriorNet(channels, base_channels)
        self.kspace_prior = FrequencyBlock(channels, base_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        image_update = self.image_prior(x)
        freq_update = self.kspace_prior(x)
        gate = self.gate(torch.cat([x, image_update, freq_update], dim=1))
        return x + gate * image_update + (1.0 - gate) * freq_update


class MDPGRecon(nn.Module):
    """MDPG-style comparison model adapted to this project's dataset format."""

    def __init__(
        self,
        config=None,
        patch_size=2,
        num_classes=2,
        base_channels=32,
        num_cascades=3,
        window_size=0,
        use_fourier=False,
        **kwargs,
    ):
        super().__init__()
        self.cascades = nn.ModuleList(
            [MDPGCascade(num_classes, base_channels) for _ in range(num_cascades)]
        )

    def forward(self, x, us_mask=None, coil_map=None):
        out = x
        for cascade in self.cascades:
            out = cascade(out)
            if us_mask is not None and coil_map is not None:
                out = _coil_data_consistency(out, x, us_mask, coil_map)
        return out


class VarNetRegularizer(nn.Module):
    def __init__(self, channels=2, base_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.GELU(),
            ResidualConvBlock(base_channels),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(base_channels * 4),
            ResidualConvBlock(base_channels * 4),
        )
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, channels, 3, padding=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.bottleneck(x2)
        x = self.up1(x, x1)
        x = self.up2(x, x0)
        return self.out(x)


class E2EVarNetCascade(nn.Module):
    def __init__(self, channels=2, base_channels=32):
        super().__init__()
        self.regularizer = VarNetRegularizer(channels, base_channels)
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.model_weight = nn.Parameter(torch.tensor(0.1))

    def soft_data_consistency(self, image, zero_fill, mask, coil_map):
        image_complex = _to_complex(image)
        zero_fill_complex = _to_complex(zero_fill)
        image_coils = image_complex.unsqueeze(1) * coil_map
        zero_fill_coils = zero_fill_complex.unsqueeze(1) * coil_map

        pred_kspace = torch.fft.fft2(image_coils, norm="ortho")
        ref_kspace = torch.fft.fft2(zero_fill_coils, norm="ortho")

        mask = mask.to(dtype=pred_kspace.real.dtype)
        while mask.dim() < pred_kspace.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred_kspace.real)

        corrected_kspace = pred_kspace - self.dc_weight * mask * (pred_kspace - ref_kspace)
        corrected_image = torch.fft.ifft2(corrected_kspace, norm="ortho")
        corrected_image = torch.sum(corrected_image * torch.conj(coil_map), dim=1)
        return _to_channels(corrected_image).type_as(image)

    def forward(self, image, zero_fill, mask, coil_map):
        regularized = image - self.model_weight * self.regularizer(image)
        return self.soft_data_consistency(regularized, zero_fill, mask, coil_map)


class E2EVarNet(nn.Module):
    """E2E-VarNet-style cascaded reconstruction adapted to this project's tensors."""

    def __init__(
        self,
        config=None,
        patch_size=2,
        num_classes=2,
        base_channels=32,
        num_cascades=6,
        window_size=0,
        use_fourier=False,
        **kwargs,
    ):
        super().__init__()
        self.cascades = nn.ModuleList(
            [E2EVarNetCascade(num_classes, base_channels) for _ in range(num_cascades)]
        )

    def forward(self, x, us_mask=None, coil_map=None):
        out = x
        for cascade in self.cascades:
            out = cascade(out, x, us_mask, coil_map)
        return out


class DHBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.frequency = FrequencyBlock(channels, channels * 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        spatial = self.local(x)
        spectral = self.frequency(x)
        return x + self.scale * self.fuse(torch.cat([spatial, spectral], dim=1))


class DHStage(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[DHBlock(channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class DHMambaRecon(nn.Module):
    """DH-Mamba-style dual-domain hierarchical comparison model."""

    def __init__(
        self,
        config=None,
        patch_size=2,
        num_classes=2,
        base_channels=32,
        depths=(2, 2, 2),
        window_size=0,
        use_fourier=True,
        **kwargs,
    ):
        super().__init__()
        self.embed = nn.Conv2d(num_classes, base_channels, 3, padding=1)
        self.stage1 = DHStage(base_channels, depths[0])
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.stage2 = DHStage(base_channels * 2, depths[1])
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.stage3 = DHStage(base_channels * 4, depths[2])
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, num_classes, 3, padding=1)

    def forward(self, x, us_mask=None, coil_map=None):
        s1 = self.stage1(self.embed(x))
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        out = self.up1(s3, s2)
        out = self.up2(out, s1)
        out = x + self.out(out)
        if us_mask is not None and coil_map is not None:
            out = _coil_data_consistency(out, x, us_mask, coil_map)
        return out
