

from tensorboardX import SummaryWriter

import torch.optim as optim

import torch.nn.functional as F

import torch

import torch.backends.cudnn as cudnn

import numpy as np

import math

import argparse

import logging

import os

import random

import shutil

import sys
import time



from torch.utils.data import DataLoader

from tqdm import tqdm


from config import get_config


from dataloaders.dataset import ixi_dataset, fastmri_dataset

from val_2D import test_single_slice


parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='PGD-Mamba')

parser.add_argument('--exp', type=str,
                    default='mamba_unrolled', help='PGD-Mamba')


parser.add_argument('--dataset', type=str,
                    default='fastmri', help='PGD-Mamba')


parser.add_argument('--model', type=str,
                    default='mamba_unrolled', help='PGD-Mamba')

parser.add_argument('--num_classes', type=int,  default=2,
                    help='PGD-Mamba')



parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='PGD-Mamba', )

parser.add_argument(
    "--opts",
    help="PGD-Mamba",
    default=None,
    nargs='+',
)

parser.add_argument('--zip', action='store_true',
                    help='PGD-Mamba')

parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='PGD-Mamba'
                    'PGD-Mamba'
                    'PGD-Mamba')

parser.add_argument('--resume', help='PGD-Mamba')

parser.add_argument('--accumulation-steps', type=int,
                    help="PGD-Mamba")

parser.add_argument('--use-checkpoint', action='store_true',
                    help="PGD-Mamba")

parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='PGD-Mamba')

parser.add_argument('--tag', help='PGD-Mamba')

parser.add_argument('--eval', action='store_true',
                    help='PGD-Mamba')

parser.add_argument('--throughput', action='store_true',
                    help='PGD-Mamba')



parser.add_argument('--max_iterations', type=int,
                    default=100000, help='PGD-Mamba')

parser.add_argument('--batch_size', type=int, default=4,
                    help='PGD-Mamba')

parser.add_argument('--deterministic', type=int,  default=1,
                    help='PGD-Mamba')

parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='PGD-Mamba')

parser.add_argument('--patch_size', type=int,  default=2,
                    help='PGD-Mamba')

parser.add_argument('--seed', type=int,  default=1337, help='PGD-Mamba')

parser.add_argument('--labeled_num', type=int, default=100,
                    help='PGD-Mamba')
# GPU ID
parser.add_argument('--gpu_id', type=int,  default=0)

parser.add_argument('--acceleration', type=int, default=4, choices=[4, 8],
                    help='PGD-Mamba')

parser.add_argument('--use_fourier', type=int, default=1,
                    help='PGD-Mamba')

parser.add_argument('--window_size', type=int, default=0,
                    help='PGD-Mamba')
parser.add_argument('--disable_lpips', action='store_true',
                    help='Disable LPIPS validation metric')


args = parser.parse_args()


print("dataset: ", args.dataset)
args.model = args.model.lower().replace("-", "_")


if args.model == "mamba_unrolled":
    
    from networks.vision_mamba import MambaUnrolled as VIM_seg
elif args.model == "mamba_unet":
    
    from networks.vision_mamba import MambaUnet as VIM_seg
elif args.model == "swin_unet":
    
    from networks.vision_transformer import SwinUnet as VIM_seg
    
    args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
elif args.model == "swin_unrolled":
    
    from networks.vision_transformer import SwinUnrolled as VIM_seg
    
    args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
elif args.model == "unet":
    
    from networks.unet import UNet as VIM_seg
elif args.model == "mdpg":
    from networks.comparison_models import MDPGRecon as VIM_seg
elif args.model in ["e2e_varnet", "e2evarnet", "varnet"]:
    args.model = "e2e_varnet"
    from networks.comparison_models import E2EVarNet as VIM_seg
elif args.model in ["dh_mamba", "dh_mamab", "dhmamba"]:
    args.model = "dh_mamba"
    from networks.comparison_models import DHMambaRecon as VIM_seg
else:
    raise ValueError(
        "Unknown model '{}'. Supported models: mamba_unrolled, mamba_unet, "
        "swin_unet, swin_unrolled, unet, e2e_varnet, mdpg, dh_mamba".format(args.model)
    )



config = get_config(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model, sample_inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis(model, sample_inputs).total() / 1e9
    except Exception:
        try:
            if hasattr(model, "flops"):
                return float(model.flops()) / 1e9
        except Exception:
            pass
    return None


def benchmark_inference_time(model, sample_inputs, warmup=10, repeat=50):
    model.eval()
    device = sample_inputs[0].device
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*sample_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(repeat):
            start = time.time()
            _ = model(*sample_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.time() - start) * 1000.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return float(np.mean(timings)), float(np.std(timings))


def adjust_learning_rate(optimizer, epoch, total_epochs=100, lr=1e-3, warmup_epochs=5, min_lr=1e-6):
    """Documentation omitted for release."""
    
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) /
             (total_epochs - warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train(args, snapshot_path):
    """Documentation omitted for release."""
    
    base_lr = args.base_lr  
    batch_size = args.batch_size  
    max_iterations = args.max_iterations  

    
    model = VIM_seg(config, patch_size=args.patch_size,
                    num_classes=2,
                    window_size=args.window_size,
                    use_fourier=bool(args.use_fourier)).cuda()
    
    
    denoiser = None
    if getattr(config.MODEL, 'USE_PNP', False):
        from networks.denoiser import UNetDenoiser
        denoiser = UNetDenoiser(
            in_channels=config.MODEL.VSSM.IN_CHANS,
            out_channels=config.MODEL.VSSM.IN_CHANS,
            base_channels=getattr(config.MODEL, 'PNP_BASE_CHANNELS', 32),
            depth=getattr(config.MODEL, 'PNP_DEPTH', 4)
        ).cuda()
        
        for p in denoiser.parameters():
            p.requires_grad = False
        denoiser.res_weight.requires_grad = True
    
    # --------------------------------------------
    
    # model.load_from(config)

    
    if args.dataset == "ixi":
        
        
        acc_suffix = "_8x" if args.acceleration == 8 else ""
        split_train = "train" + acc_suffix
        split_val = "val" + acc_suffix
        
        print(f"--- Loading IXI {args.acceleration}x dataset ---")
        print(f"Train split: ixi_{split_train}.h5")
        print(f"Val split: ixi_{split_val}.h5")
        
        db_train = ixi_dataset(split=split_train)
        db_val = ixi_dataset(split=split_val)

    elif args.dataset == "fastmri":
        
        print(f"--- Loading FastMRI {args.acceleration}x dataset ---")
        train_split = "train" if args.acceleration == 4 else f"train_{args.acceleration}x"
        val_split = "val" if args.acceleration == 4 else f"val_{args.acceleration}x"
        db_train = fastmri_dataset(split=train_split, acceleration=args.acceleration)
        db_val = fastmri_dataset(split=val_split, acceleration=args.acceleration)

    def worker_init_fn(worker_id):
        """Documentation omitted for release."""
        random.seed(args.seed + worker_id)

    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    # Log efficiency on one validation-size slice.
    try:
        profile_batch = next(iter(valloader))
        profile_inputs = (
            profile_batch['us_image'].cuda(),
            profile_batch['us_mask'].cuda(),
            profile_batch['coil_map'].cuda(),
        )
        params_m = count_parameters(model)
        flops_g = estimate_flops(model, profile_inputs)
        time_ms, time_std_ms = benchmark_inference_time(model, profile_inputs)
        logging.info('Requested model: {}'.format(args.model))
        logging.info('Instantiated model class: {}'.format(model.__class__.__name__))
        logging.info('Params (M): %.4f' % params_m)
        logging.info('FLOPs (G): %.4f' % flops_g if flops_g is not None else 'FLOPs (G): unavailable')
        logging.info('Time (ms): %.4f +/- %.4f' % (time_ms, time_std_ms))
    except Exception as e:
        logging.info('Efficiency profiling failed: {}'.format(e))

    
    model.train()

    
    train_params = list(model.parameters())
    if denoiser is not None:
        train_params += [denoiser.res_weight]
    optimizer = optim.AdamW(train_params, lr=base_lr)

    
    writer = SummaryWriter(snapshot_path + '/log')
    lpips_metric = None
    if not args.disable_lpips:
        try:
            from networks.lpips import LPIPS
            lpips_metric = LPIPS().cuda()
        except Exception as e:
            logging.info('LPIPS metric disabled: {}'.format(e))
    
    logging.info("{} iterations per epoch".format(len(trainloader)))

    
    iter_num = 0
    
    max_epoch = max_iterations // len(trainloader) + 1
    
    best_performance = 0.0
    best_ssim_at_best_psnr = 0.0
    best_lpips_at_best_psnr = 0.0
    best_iter = 0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            
            us_image, fs_target, us_mask, coil_maps = sampled_batch['us_image'], sampled_batch[
                'fs_image'], sampled_batch['us_mask'], sampled_batch['coil_map']
            
            us_image, fs_target, us_mask, coil_maps = us_image.cuda(
            ), fs_target.cuda(), us_mask.cuda(), coil_maps.cuda()

            
            outputs = model(us_image, us_mask, coil_maps)

            
            if denoiser is not None and iter_num > config.MODEL.PNP_WARMUP_ITER:
                outputs = denoiser(outputs)
                
                
                res = torch.fft.fft2(outputs[:, 0, :, :] + 1j * outputs[:, 1, :, :], norm='ortho')
                res = res.unsqueeze(1) * coil_maps
                
                
                
                us_kspace = torch.fft.fft2(us_image[:, 0, :, :] + 1j * us_image[:, 1, :, :], norm='ortho')
                res = (1 - us_mask) * res + us_mask * us_kspace.unsqueeze(1)
                
                outputs_dc = torch.fft.ifft2(torch.sum(res * torch.conj(coil_maps), dim=1), norm='ortho')
                outputs = torch.stack([outputs_dc.real, outputs_dc.imag], dim=1)
            # ----------------------------------------

            
            outputs = torch.abs(outputs[:, 0, :, :] + 1j * outputs[:, 1, :, :])
            
            loss = torch.mean(torch.abs(outputs - fs_target[:, 0, :, :]))

            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            
            epoch_float = epoch_num + (i_batch / len(trainloader))
            lr_ = adjust_learning_rate(optimizer, epoch_float, total_epochs=max_epoch, lr=args.base_lr)

            
            iter_num = iter_num + 1
            
            writer.add_scalar('info/lr', lr_, iter_num)
            
            writer.add_scalar('info/total_loss', loss, iter_num)

            
            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))

            
            if iter_num % 250 == 0:
                
                image_ = torch.abs(
                    us_image[0, 0, :, :] + 1j * us_image[0, 1, :, :]).unsqueeze(0)
                
                writer.add_image('train/Image', image_, iter_num)
                
                image = outputs[0].unsqueeze(0)
                writer.add_image('train/Prediction', image, iter_num)
                
                labs = fs_target[0]
                writer.add_image('train/GroundTruth', labs, iter_num)

            
            if iter_num > 0 and iter_num % 250 == 0:
                
                model.eval()
                
                metric_list = 0.0
                
                for i_batch, sampled_batch in enumerate(valloader):
                    
                    metric_i = test_single_slice(
                        sampled_batch['us_image'].cuda(), sampled_batch['fs_image'].cuda(), sampled_batch['us_mask'].cuda(), sampled_batch['coil_map'].cuda(), model, denoiser=denoiser, lpips_metric=lpips_metric)
                    
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                
                writer.add_scalar('info/val_{}_psnr'.format(i_batch),
                                  metric_list[0, 0], iter_num)
                writer.add_scalar('info/val_{}_ssim'.format(i_batch),
                                  metric_list[0, 1], iter_num)
                writer.add_scalar('info/val_{}_lpips'.format(i_batch),
                                  metric_list[0, 2], iter_num)

                
                mean_psnr = np.nanmean(metric_list, axis=0)[0]
                mean_ssim = np.nanmean(metric_list, axis=0)[1]
                mean_lpips = np.nanmean(metric_list, axis=0)[2]

                
                writer.add_scalar('info/val_mean_psnr', mean_psnr, iter_num)
                writer.add_scalar('info/val_mean_ssim', mean_ssim, iter_num)
                writer.add_scalar('info/val_mean_lpips', mean_lpips, iter_num)

                
                if mean_psnr > best_performance:
                    best_performance = mean_psnr
                    best_ssim_at_best_psnr = mean_ssim
                    best_lpips_at_best_psnr = mean_lpips
                    best_iter = iter_num
                    
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_psnr_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                
                logging.info(
                    'iteration %d : mean_psnr : %f mean_ssim : %f mean_lpips : %f' % (iter_num, mean_psnr, mean_ssim, mean_lpips))
                
                model.train()

            
            if iter_num % 250 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    writer.close()
    
    
    logging.info('-------------------------------------------')
    logging.info('Training Finished!')
    logging.info('Best Performance at iteration %d:' % best_iter)
    logging.info('Best Mean PSNR: %f' % best_performance)
    logging.info('Corresponding Mean SSIM: %f' % best_ssim_at_best_psnr)
    logging.info('Corresponding Mean LPIPS: %f' % best_lpips_at_best_psnr)
    logging.info('-------------------------------------------')
    
    return "Training Finished!"


if __name__ == "__main__":
    """Documentation omitted for release."""
    
    if not args.deterministic:
        
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        
        cudnn.benchmark = False
        cudnn.deterministic = True

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    torch.cuda.set_device(int(args.gpu_id))
    
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    
    backup_ignore = shutil.ignore_patterns(
        '.git', '__pycache__',
        'datasets', 'dataset', 'data',
        'model', 'models', 'checkpoints', 'runs',
        '*.h5', '*.hdf5', '*.nii', '*.nii.gz', '*.mat',
        '*.pth', '*.pt', '*.ckpt'
    )
    shutil.copytree('.', snapshot_path + '/code', ignore=backup_ignore)

    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info(str(args))
    
    train(args, snapshot_path)
