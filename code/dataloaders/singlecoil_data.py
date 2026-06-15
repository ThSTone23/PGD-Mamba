import numpy as np
import h5py 
import os


def fft2c_np(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im, axes=[-1,-2]), norm="ortho"), axes=[-1,-2]) 

def ifft2c_np(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d, axes=[-1,-2]), norm="ortho"), axes=[-1,-2])

def get_ixi_dataset(phase="train"):
    target_file = os.path.join("datasets", "ixi", "ixi_" + phase + ".h5")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Could not find dataset file: {target_file}")

    with h5py.File(target_file, "r") as f:
        raw_fs = np.array(f["data_fs"])
        raw_us = np.array(f["data_us"])
        raw_masks = np.array(f["us_masks"])

    if raw_fs.ndim == 3:
        data_fs = raw_fs[:, None, :, :]
    elif raw_fs.ndim == 4 and raw_fs.shape[1] == 1:
        data_fs = raw_fs
    else:
        raise ValueError(f"Unsupported IXI data_fs shape: {raw_fs.shape}")
    data_fs = data_fs.astype(np.float32)

    if raw_us.ndim == 4 and raw_us.shape[0] == 2:
        magnitude = raw_us[0]
        phase_map = raw_us[1] * 2 * np.pi - np.pi
        data_us_complex = magnitude * np.exp(1j * phase_map)
    elif raw_us.ndim == 4 and raw_us.shape[1] == 2:
        data_us_complex = raw_us[:, 0] + 1j * raw_us[:, 1]
    else:
        raise ValueError(f"Unsupported IXI data_us shape: {raw_us.shape}")
    data_us = np.stack([data_us_complex.real, data_us_complex.imag], axis=1).astype(np.float32)

    if raw_masks.ndim == 3:
        data_masks = raw_masks[:, None, :, :]
    elif raw_masks.ndim == 4 and raw_masks.shape[1] == 1:
        data_masks = raw_masks
    else:
        raise ValueError(f"Unsupported IXI us_masks shape: {raw_masks.shape}")
    data_masks = data_masks.astype(np.float32)

    print(f"IXI {phase}: {len(data_us)} images loaded from {target_file}")
    return data_us, data_fs, data_masks
