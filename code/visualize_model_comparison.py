import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from config import get_config
from dataloaders.dataset import fastmri_dataset, ixi_dataset


plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


MODEL_ALIASES = {
    "unet": "unet",
    "u_net": "unet",
    "e2e_varnet": "e2e_varnet",
    "e2evarnet": "e2e_varnet",
    "varnet": "e2e_varnet",
    "mamba_unet": "mamba_unet",
    "recurrent_varnet": "mamba_unet",
    "swinunet": "swin_unet",
    "swin_unet": "swin_unet",
    "swinmr": "swin_unrolled",
    "swin_unrolled": "swin_unrolled",
    "mambarecon": "mamba_unrolled",
    "mamba_recon": "mamba_unrolled",
    "mamba_unrolled": "mamba_unrolled",
    "ours": "mamba_unrolled",
    "mdpg": "mdpg",
    "dhmamba": "dh_mamba",
    "dh_mamba": "dh_mamba",
    "dh_mamab": "dh_mamba",
}


def normalize_name(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def infer_model_arch(label, path):
    label_key = normalize_name(label)
    if label_key in MODEL_ALIASES:
        return MODEL_ALIASES[label_key]

    path_key = normalize_name(path.replace(os.sep, "_"))
    for key, arch in sorted(MODEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if key in path_key:
            return arch

    raise ValueError(
        "Cannot infer model architecture for label '{}'. Use label@arch=path, "
        "for example Recurrent_Varnet@mamba_unet=/path/to/ckpt.pth".format(label)
    )


def build_model(name, config, args):
    name = normalize_name(name)
    if name == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as Model
    elif name == "mamba_unet":
        from networks.vision_mamba import MambaUnet as Model
    elif name == "swin_unet":
        from networks.vision_transformer import SwinUnet as Model
    elif name == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as Model
    elif name == "unet":
        from networks.unet import UNet as Model
    elif name == "mdpg":
        from networks.comparison_models import MDPGRecon as Model
    elif name in ["e2e_varnet", "e2evarnet", "varnet"]:
        from networks.comparison_models import E2EVarNet as Model
    elif name in ["dh_mamba", "dh_mamab", "dhmamba"]:
        from networks.comparison_models import DHMambaRecon as Model
    else:
        raise ValueError(f"Unsupported model: {name}")

    return Model(
        config,
        patch_size=args.patch_size,
        num_classes=2,
        window_size=args.window_size,
        use_fourier=bool(args.use_fourier),
    ).cuda()


def parse_checkpoints(items):
    checkpoints = []
    for item in items:
        if "=" not in item:
            raise ValueError("Checkpoint must use label=path or label@arch=path format: " + item)
        left, path = item.split("=", 1)
        left = left.strip()
        path = path.strip()
        if "@" in left:
            label, arch = left.split("@", 1)
            label = label.strip()
            arch = infer_model_arch(arch.strip(), path)
        else:
            label = left
            arch = infer_model_arch(label, path)
        checkpoints.append((label, arch, path))
    return checkpoints


def load_dataset(args):
    if args.dataset == "fastmri":
        split = args.split
        if args.acceleration != 4 and not split.endswith(f"_{args.acceleration}x"):
            split = f"{split}_{args.acceleration}x"
        print(f"Requested FastMRI split={split}, acceleration={args.acceleration}x")
        return fastmri_dataset(split=split, acceleration=args.acceleration)
    if args.dataset == "ixi":
        split = args.split
        if args.acceleration == 8 and not split.endswith("_8x"):
            split = split + "_8x"
        return ixi_dataset(split=split)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def to_magnitude(x):
    return torch.abs(x[:, 0] + 1j * x[:, 1])


def normalize_for_display(x, vmax=None):
    x = np.asarray(x)
    if vmax is None:
        vmax = np.percentile(x, 99.5)
    return np.clip(x / (vmax + 1e-8), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Use label=path or label@arch=path. Repeat this option for multiple models.")
    parser.add_argument("--dataset", type=str, default="fastmri", choices=["fastmri", "ixi"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="model_comparison.png")
    parser.add_argument("--image_output", type=str, default=None,
                        help="Optional raster image output, e.g. model_comparison.png.")
    parser.add_argument("--pdf_output", type=str, default=None,
                        help="Optional PDF output for LaTeX.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--error_max", type=float, default=None,
                        help="Upper limit for error-map colorbar. Defaults to the 99th percentile.")
    parser.add_argument("--acceleration", type=int, default=4, choices=[4, 8])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--use_fourier", type=int, default=1)

    # Required by get_config.
    parser.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="no")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O0")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    config = get_config(args)
    checkpoints = parse_checkpoints(args.checkpoint)
    dataset = load_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample = None
    for i, batch in enumerate(loader):
        if i == args.sample_idx:
            sample = batch
            break
    if sample is None:
        raise IndexError(f"sample_idx {args.sample_idx} is out of range")

    image = sample["us_image"].cuda()
    target = sample["fs_image"].cuda()
    mask = sample["us_mask"].cuda()
    coil_map = sample["coil_map"].cuda()

    zero_fill = to_magnitude(image).cpu().numpy()[0]
    gt = target.cpu().numpy()[0, 0]
    vmax = np.percentile(gt, 99.5)

    names = ["Zero-fill"]
    recons = [zero_fill]
    for display_name, model_arch, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        print(f"Loading {display_name} as {model_arch}: {ckpt_path}")
        model = build_model(model_arch, config, args)
        state = torch.load(ckpt_path, map_location="cuda:0")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            pred = model(image, mask, coil_map)
            pred = to_magnitude(pred).cpu().numpy()[0]
        names.append(display_name)
        recons.append(pred)
        del model

    names.append("Target")
    recons.append(gt)

    h, w = gt.shape
    roi = min(72, h // 3, w // 3)
    cy, cx = h // 2 + h // 8, w // 2
    y1 = max(0, min(h - roi, cy - roi // 2))
    x1 = max(0, min(w - roi, cx - roi // 2))
    y2, x2 = y1 + roi, x1 + roi

    cols = len(recons)
    fig, axes = plt.subplots(3, cols, figsize=(2.4 * cols, 7.2), facecolor="black")
    if args.error_max is None:
        error_max = np.percentile([np.abs(gt - r) for r in recons[:-1]], 99.0)
    else:
        error_max = args.error_max
    error_max = max(float(error_max), 1e-8)

    for j, (name, img) in enumerate(zip(names, recons)):
        zoom = normalize_for_display(img[y1:y2, x1:x2], vmax)
        axes[0, j].imshow(zoom, cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")

        axes[1, j].imshow(normalize_for_display(img, vmax), cmap="gray", vmin=0, vmax=1)
        axes[1, j].add_patch(Rectangle((x1, y1), roi, roi, linewidth=1.5,
                                       edgecolor="orangered", facecolor="none", linestyle=":"))
        axes[1, j].axis("off")

        axes[2, j].axis("off")
        if name != "Target":
            error = np.abs(gt - img)
            axes[2, j].imshow(error, cmap="hot", vmin=0, vmax=error_max)

    legend_ax = axes[2, -1]
    legend_ax.set_facecolor("black")
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=error_max), cmap="hot")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=legend_ax, fraction=0.28, pad=0.08)
    cbar.set_ticks([0, error_max])
    cbar.set_ticklabels(["0", f"{error_max:.3g}"])
    cbar.set_label("Error", color="white", rotation=90, labelpad=10, fontsize=12)
    cbar.ax.tick_params(colors="white", labelsize=10, length=0)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.8)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout(pad=0.5)

    outputs = [args.output]
    if args.image_output is not None:
        outputs.append(args.image_output)
    if args.pdf_output is not None:
        outputs.append(args.pdf_output)

    seen = set()
    for output_path in outputs:
        if output_path in seen:
            continue
        seen.add(output_path)
        plt.savefig(output_path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
        print(f"Saved comparison figure to {output_path}")


if __name__ == "__main__":
    main()
