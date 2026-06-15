import argparse
import os

import h5py
import numpy as np


def _center_square_mask(height, width, center_fraction):
    center_pixels = max(1, int(round(height * width * center_fraction)))
    side = max(1, int(round(np.sqrt(center_pixels))))
    side_h = min(height, side)
    side_w = min(width, side)
    top = (height - side_h) // 2
    left = (width - side_w) // 2

    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + side_h, left:left + side_w] = True
    return mask


def _subset_2d_mask(mask_4x, acceleration, center_fraction, rng):
    if mask_4x.ndim == 3:
        mask_4x = mask_4x[0]

    height, width = mask_4x.shape
    source = mask_4x > 0.5
    source_indices = np.argwhere(source)
    if len(source_indices) == 0:
        raise ValueError("Input mask has no sampled k-space points.")

    target_points = max(1, int(round(height * width / float(acceleration))))
    center_region = _center_square_mask(height, width, center_fraction)
    selected = source & center_region

    if int(selected.sum()) > target_points:
        center_points = np.argwhere(selected)
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        dist = (center_points[:, 0] - cy) ** 2 + (center_points[:, 1] - cx) ** 2
        keep = center_points[np.argsort(dist)[:target_points]]
        selected = np.zeros_like(source, dtype=bool)
        selected[keep[:, 0], keep[:, 1]] = True
        return selected.astype(np.float32)

    remaining_needed = target_points - int(selected.sum())
    candidates = np.argwhere(source & (~selected))

    if remaining_needed > 0 and len(candidates) > 0:
        chosen_ids = rng.choice(
            len(candidates),
            size=min(remaining_needed, len(candidates)),
            replace=False,
        )
        chosen = candidates[chosen_ids]
        selected[chosen[:, 0], chosen[:, 1]] = True

    return selected.astype(np.float32)


def make_8x_masks(masks_4x, acceleration=8, center_fraction=0.04, seed=1337):
    rng = np.random.RandomState(seed)
    masks_4x = np.asarray(masks_4x)
    if masks_4x.ndim == 4 and masks_4x.shape[1] == 1:
        source = masks_4x[:, 0]
        add_channel = True
    elif masks_4x.ndim == 3:
        source = masks_4x
        add_channel = False
    else:
        raise ValueError(f"Unsupported mask shape: {masks_4x.shape}")

    masks_8x = np.stack(
        [_subset_2d_mask(mask, acceleration, center_fraction, rng) for mask in source],
        axis=0,
    )
    if add_channel:
        masks_8x = masks_8x[:, None]
    return masks_8x.astype(np.float32)


def copy_group_or_dataset(src, dst, key, compression):
    obj = src[key]
    if isinstance(obj, h5py.Group):
        out_group = dst.create_group(key)
        for subkey in obj.keys():
            out_group.create_dataset(
                subkey,
                data=obj[subkey][()],
                compression=compression,
            )
    else:
        dst.create_dataset(key, data=obj[()], compression=compression)


def convert_file(input_path, output_path, acceleration, center_fraction, seed, compression):
    print(f"Converting {input_path}")
    with h5py.File(input_path, "r") as src:
        masks_4x = src["us_masks"][()]
        masks_8x = make_8x_masks(
            masks_4x,
            acceleration=acceleration,
            center_fraction=center_fraction,
            seed=seed,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as dst:
            for key in src.keys():
                if key == "us_masks":
                    continue
                copy_group_or_dataset(src, dst, key, compression)
            dst.create_dataset("us_masks", data=masks_8x, compression=compression)
            dst.attrs["source_file"] = os.path.basename(input_path)
            dst.attrs["acceleration"] = acceleration
            dst.attrs["center_fraction"] = center_fraction

    print(f"Saved {output_path}")
    print(f"4x mask ratio: {float(np.mean(masks_4x)):.4f}")
    print(f"{acceleration}x mask ratio: {float(np.mean(masks_8x)):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets/fastmri")
    parser.add_argument("--output_dir", type=str, default="datasets/fastmri")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--center_fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    for offset, split in enumerate(args.splits):
        input_path = os.path.join(args.input_dir, f"fastmri_{split}.h5")
        output_path = os.path.join(args.output_dir, f"fastmri_{split}_{args.acceleration}x.h5")
        convert_file(
            input_path,
            output_path,
            acceleration=args.acceleration,
            center_fraction=args.center_fraction,
            seed=args.seed + offset,
            compression=compression,
        )


if __name__ == "__main__":
    main()
