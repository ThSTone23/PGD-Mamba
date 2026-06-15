"""Documentation omitted for release."""

import numpy as np
import h5py
import os


def fft2c_np(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im, axes=[-1,-2]), norm="ortho"), axes=[-1,-2])

def ifft2c_np(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d, axes=[-1,-2]), norm="ortho"), axes=[-1,-2])


def get_ixi_mc_dataset(phase="train", acceleration=4):
    """Documentation omitted for release."""
    acc_suffix = f"_{acceleration}x" if acceleration != 4 else ""
    target_file = f"datasets/ixi_mc/ixi_mc_{phase}{acc_suffix}.h5"

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Could not find dataset file: {target_file}")

    f = h5py.File(target_file, 'r')

    
    # data_fs: [N, 256, 256] -> [N, 1, 256, 256]
    raw_fs = np.array(f['data_fs'])  # [N, H, W]
    data_fs = np.expand_dims(np.transpose(raw_fs, (0, 2, 1)), axis=1)  
    data_fs = data_fs.astype(np.float32)

    
    
    data_us = np.transpose(np.array(f['data_us']), (1, 0, 3, 2))  # [N, 2, W, H]
    phase_ = (data_us[:, 1, :, :] * 2 * np.pi) - np.pi
    data_us = data_us[:, 0, :, :] * np.exp(1j * phase_)
    us_array = np.stack([np.real(data_us), np.imag(data_us)], axis=1).astype(np.float32)  # [N, 2, H, W]

    
    raw_masks = np.array(f['us_masks'])  # [N, H, W]
    data_masks = np.expand_dims(np.transpose(raw_masks, (0, 2, 1)), axis=1)  # [N, 1, W, H]
    data_masks = data_masks.astype(np.float32)

    
    raw_ref = np.array(f['ref_fs'])  # [N, H, W]
    data_ref = np.expand_dims(np.transpose(raw_ref, (0, 2, 1)), axis=1)  # [N, 1, W, H]
    data_ref = data_ref.astype(np.float32)

    f.close()

    print(f"Loaded IXI multicontrast ({phase}): {len(us_array)} slices")
    print(f"  us_image: {us_array.shape}, fs_image: {data_fs.shape}")
    print(f"  mask: {data_masks.shape}, ref_image: {data_ref.shape}")

    return us_array, data_fs, data_masks, data_ref
