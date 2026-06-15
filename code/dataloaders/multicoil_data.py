import numpy as np
import h5py
import os

                    
def fft2c_np(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im, axes=[-1,-2]), axes=[-1,-2]), axes=[-1,-2]) 

def ifft2c_np(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d, axes=[-1,-2]), axes=[-1,-2]), axes=[-1,-2])


def get_fastmri_dataset(phase='train', acceleration=4):
    base_phase = phase.replace('_8x', '').replace('_4x', '')
    if acceleration == 4:
        candidates = [
            os.path.join('datasets', 'fastmri', 'fastmri_' + phase + '.h5'),
            os.path.join('datasets', 'fastmri', 'fastmri_' + base_phase + '.h5'),
            os.path.join('datasets', 'fastmri_' + phase + '.h5'),
            os.path.join('datasets', 'fastmri_' + base_phase + '.h5'),
        ]
    else:
        preferred_phase = phase if phase.endswith(f'_{acceleration}x') else f'{base_phase}_{acceleration}x'
        alt_phase = f'{acceleration}x_{base_phase}'
        candidates = [
            os.path.join('datasets', 'fastmri', 'fastmri_' + preferred_phase + '.h5'),
            os.path.join('datasets', 'fastmri', 'fastmri_' + alt_phase + '.h5'),
            os.path.join('datasets', 'fastmri_' + preferred_phase + '.h5'),
            os.path.join('datasets', 'fastmri_' + alt_phase + '.h5'),
        ]
    target_file = next((path for path in candidates if os.path.exists(path)), candidates[0])
    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"Could not find FastMRI {acceleration}x dataset file. Tried: {candidates}. "
            f"Generate it first with convert_fastmri_4x_to_8x.py."
        )
    data_fs=LoadDataSetMultiCoil(target_file, 'images_fs', padding = False, Norm = True, channel_cat = False)
    masks_dummy = LoadDataSetMultiCoil(target_file, 'us_masks', padding = False, Norm = False, channel_cat = False, clip=False)
    coil_maps_dummy=LoadDataSetMultiCoil(target_file, 'coil_maps', padding = False, Norm = False, channel_cat = False)  
    masks_dummy = np.real(masks_dummy).astype(np.float32)
    print(f"FastMRI {base_phase}: loaded {target_file}, sampled ratio={masks_dummy.mean():.4f}")

    us_image_dummy = np.sum(ifft2c_np(fft2c_np(np.tile(data_fs, [1,5,1,1]) * coil_maps_dummy) * np.tile(masks_dummy,[1,5,1,1])) * np.conj(coil_maps_dummy), axis=1) 
    us_image_dummy = np.stack([np.real(us_image_dummy), np.imag(us_image_dummy)], axis=1)
    masks=masks_dummy                 
    coil_maps=coil_maps_dummy 
    data_us = us_image_dummy

    data_fs = np.abs(np.real(data_fs) + 1j * np.imag(data_fs))
    data_fs = data_fs.astype(np.float32)
    data_us = data_us.astype(np.float32)
    masks = masks.astype(np.float32)

    return data_us, data_fs, masks, coil_maps


def LoadDataSetMultiCoil(load_dir, variable='images_fs', clip=True, padding=False, Norm=True, res=[256, 256], slices=10, is_complex=True, channel_cat=False):
    with h5py.File(load_dir, 'r') as f:
        if variable in f:
            group = f[variable]
            if isinstance(group, h5py.Group): 
                real_part = np.array(group['real'], dtype=np.float32) 
                imag_part = np.array(group['imag'], dtype=np.float32)
                data = real_part + 1j * imag_part 
            else:
                data = np.array(group, dtype=np.float32)

    data = np.asarray(data, dtype=np.complex64)

    if data.ndim == 3:
        data = np.expand_dims(np.transpose(data, (0, 1, 2)), axis=1)  
    else:
        data = np.transpose(data, (1, 0, 2, 3)) 

    if Norm:
        max_vals = np.abs(data).max(axis=(2, 3), keepdims=True)
        max_vals[max_vals == 0] = 1  
        data /= max_vals 

    if channel_cat:
        data = np.concatenate((data.real, data.imag), axis=1)

    if clip:
        data = data[:,:, int((data.shape[2] - res[0]) / 2): int(data.shape[2] - (data.shape[2] - res[0]) / 2),  int((data.shape[3] - res[1]) / 2): int(data.shape[3] - (data.shape[3] - res[1]) / 2) ]


    return data
