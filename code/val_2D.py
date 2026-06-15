"""Documentation omitted for release."""

import numpy as np
import torch
from utils import psnr


def calculate_psnr_ssim(pred, gt):
    """Documentation omitted for release."""
    psnr_, ssim_ = psnr.compute_psnr(pred, gt), psnr.compute_ssim(pred, gt)
    if np.isnan(psnr_) or np.isnan(ssim_):
        return 0, 0
    else:
        return psnr_, ssim_


def test_single_slice(image, target, us_mask, coil_map, net, denoiser=None, lpips_metric=None):
    """Documentation omitted for release."""
    target_tensor = target

    net.eval()
    with torch.no_grad():
        out = net(image, us_mask, coil_map)
        
        if denoiser is not None:
            out = denoiser(out)
        out_mag = torch.abs(out[:, 0, :, :] + 1j * out[:, 1, :, :])
        lpips_value = 0.0
        if lpips_metric is not None:
            out_lpips = out_mag.unsqueeze(1).repeat(1, 3, 1, 1)
            target_lpips = target_tensor[:, 0:1, :, :].repeat(1, 3, 1, 1)
            try:
                lpips_value = lpips_metric(
                    out_lpips, target_lpips
                ).item()
            except Exception as e:
                print(f"LPIPS metric failed and was skipped for this slice: {e}")
                lpips_value = np.nan

    target = target_tensor.cpu().detach().numpy()
    out = out_mag.cpu().detach().numpy()
    metric_list = []
    psnr_, ssim_ = calculate_psnr_ssim(out[0, :, :], target[0, 0, :, :])
    metric_list.append((psnr_, ssim_, lpips_value))
    return metric_list
