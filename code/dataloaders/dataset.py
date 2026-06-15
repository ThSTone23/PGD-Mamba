import numpy as np
from torch.utils.data import Dataset
from scipy import ndimage
from dataloaders import singlecoil_data, multicoil_data

try:
    from dataloaders import multicontrast_data
except ImportError:
    multicontrast_data = None


class fastmri_dataset(Dataset):
    def __init__(self, split="train", acceleration=4):
        self.us_images, self.fs_images, self.us_masks, self.coil_maps = multicoil_data.get_fastmri_dataset(
            split, acceleration=acceleration
        )

    def __len__(self):
        return len(self.fs_images)

    def __getitem__(self, idx):
        us_im = self.us_images[idx]
        fs_im = self.fs_images[idx]
        us_mask = self.us_masks[idx]
        coil_map = self.coil_maps[idx]
        return dict(us_image=us_im, fs_image=fs_im, us_mask=us_mask, coil_map=coil_map)


class ixi_dataset(Dataset):
    def __init__(self, split="train"):
        # Allow split values such as "train_8x" and "val_8x".
        self.us_images, self.fs_images, self.us_masks = singlecoil_data.get_ixi_dataset(split)
        self.coil_map = np.ones([1, 256, 256], dtype=np.float32)

    def __len__(self):
        return len(self.fs_images)

    def __getitem__(self, idx):
        us_im = self.us_images[idx]
        fs_im = self.fs_images[idx]
        us_mask = self.us_masks[idx]
        return dict(us_image=us_im, fs_image=fs_im, us_mask=us_mask, coil_map=self.coil_map)


class ixi_mc_dataset(Dataset):
    """IXI multi-contrast paired dataset: undersampled T2 reconstruction with T1 reference."""

    def __init__(self, split="train", acceleration=4):
        if multicontrast_data is None:
            raise ImportError("multicontrast_data is required for ixi_mc_dataset but was not found.")
        self.us_images, self.fs_images, self.us_masks, self.ref_images = (
            multicontrast_data.get_ixi_mc_dataset(split, acceleration)
        )
        self.coil_map = np.ones([1, 256, 256], dtype=np.float32)

    def __len__(self):
        return len(self.fs_images)

    def __getitem__(self, idx):
        us_im = self.us_images[idx]
        fs_im = self.fs_images[idx]
        us_mask = self.us_masks[idx]
        ref_im = self.ref_images[idx]
        return dict(
            us_image=us_im,
            fs_image=fs_im,
            us_mask=us_mask,
            coil_map=self.coil_map,
            ref_image=ref_im,
        )


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
