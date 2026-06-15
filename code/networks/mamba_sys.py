"""Documentation omitted for release."""

import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.fft
from einops import rearrange, repeat

def polar_sorting(tensor):
    """Documentation omitted for release."""
    B, C, H, W = tensor.shape
    center_y, center_x = H // 2, W // 2
    y = torch.arange(H, device=tensor.device) - center_y
    x = torch.arange(W, device=tensor.device) - center_x
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    angle = torch.atan2(grid_y, grid_x)
    sort_key = distance + (angle + math.pi) / (2 * math.pi * max(H, W))
    sorted_indices = torch.argsort(sort_key.view(-1))
    tensor_flat = tensor.view(B, C, H * W)
    return tensor_flat[:, :, sorted_indices], sorted_indices

def polar_restore(one_d_sequence, indices, shape):
    """Documentation omitted for release."""
    B, C, L = one_d_sequence.shape
    H, W = shape
    restored = torch.zeros((B, C, H * W), device=one_d_sequence.device, dtype=one_d_sequence.dtype)
    restored.scatter_(2, indices.expand(B, C, -1), one_d_sequence)
    return restored.view(B, C, H, W)

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attention(x)

class HybridGate(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Conv2d(dim, dim * 2, 1)
        self.ca = ChannelAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.expand(x)
        x1, x2 = x.chunk(2, dim=1)
        
        x1 = self.ca(x1)
        
        x2 = self.mlp(x2.permute(0, 2, 3, 1).reshape(-1, C)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x1 * x2 

from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

try:
    from taming.models.vqgan import VQModel
except ImportError:
    VQModel = None





class PatchEmbed2D(nn.Module):
    r"""Documentation omitted for release."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        
        
        if patch_size[0] == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
            )
        
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        
        x = self.proj(x) # [B, C, H/k, W/k]
        
        
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r"""Documentation omitted for release."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        
        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :]  
        x3 = x[:, 1::2, 1::2, :]  

        
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2 W/2 2*C

        return x

class PatchExpand(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        
        x = self.expand(x)
        B, H, W, C = x.shape
        
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        
        x = self.expand(x)
        B, H, W, C = x.shape
        
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)

        return x
    

class Unpatchify(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        
        self.layer = nn.Linear(dim, 2 * dim_scale**2, bias=False)

    def forward(self, x):
        
        x = self.layer(x)
        _, _, _, C = x.shape
        
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        
        return x.permute(0, 3, 1, 2)

class SS2D(nn.Module):
    """Documentation omitted for release."""
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        window_size=0,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.window_size = window_size
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  
        
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()  

        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        
        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """Documentation omitted for release."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """Documentation omitted for release."""
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  
        if copies > 1:
            
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """Documentation omitted for release."""
        
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  
        
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        """Documentation omitted for release."""
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W  
        K = 4  

        
        
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  
        
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_core_windowed(self, x: torch.Tensor, window_size: int):
        B, C, H, W = x.shape
        # Padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        Hp, Wp = x.shape[2], x.shape[3]
        
        # Partition
        x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        
        # Scan
        y = self.forward_core(x) # [B*nh*nw, window_size, window_size, C]
        
        # Merge
        y = y.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        y = y.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, Hp, Wp)
        
        # Unpad
        if pad_h > 0 or pad_w > 0:
            y = y[:, :, :H, :W]
        
        return y.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor, **kwargs):
        """Documentation omitted for release."""
        B, H, W, C = x.shape

        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        
        if self.window_size > 0:
            y = self.forward_core_windowed(x, self.window_size)
        else:
            y = self.forward_core(x)
        
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out




class GSFF(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim):
        super(GSFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_s, x_f):
        
        
        B, H, W, C = x_s.shape
        
        
        y = self.avg_pool(x_s.permute(0, 3, 1, 2)).view(B, C)
        gate = self.mlp(y).view(B, 1, 1, C)
        
        
        return gate * x_s + (1 - gate) * x_f


class InceptionLocalMixer(nn.Module):
    """Local branch of Inception-VSSM: 3x3 and orthogonal large-band DWConvs."""

    def __init__(self, dim):
        super().__init__()
        c1 = dim // 3
        c2 = dim // 3
        c3 = dim - c1 - c2
        self.splits = (c1, c2, c3)
        self.square = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1)
        self.band_h = nn.Conv2d(c2, c2, kernel_size=(3, 11), padding=(1, 5), groups=c2)
        self.band_v = nn.Conv2d(c2, c2, kernel_size=(11, 3), padding=(5, 1), groups=c2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        x_square, x_band, x_id = torch.split(x_nchw, self.splits, dim=1)
        x_square = self.square(x_square)
        x_band = self.band_h(x_band) + self.band_v(x_band)
        x_local = torch.cat([x_square, x_band, x_id], dim=1)
        x_local = x_local.permute(0, 2, 3, 1).contiguous()
        return self.proj(x_local)


class VSSBlock(nn.Module):
    """Documentation omitted for release."""
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        window_size=0,
        use_fourier=True, 
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.use_fourier = use_fourier
        
        
        self.inception_local = InceptionLocalMixer(hidden_dim)
        self.spatial_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, window_size=window_size, **kwargs)
        
        
        if self.use_fourier:
            self.fourier_mamba = SS2D(d_model=hidden_dim, d_state=d_state, **kwargs)
            self.fourier_norm = norm_layer(hidden_dim)
            self.gsff = GSFF(hidden_dim) 

        
        self.lem = HybridGate(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input: torch.Tensor, z_prior=None, lcma=None):
        if input.dim() == 4:
            B, H, W, C = input.shape
            L = H * W
            is_4d = True
        else:
            B, L, C = input.shape
            H = W = int(L**0.5)
            is_4d = False
        
        
        x = self.ln_1(input) # [B, H, W, C] or [B, L, C]
        if not is_4d:
            x_spatial = x.view(B, H, W, C)
        else:
            x_spatial = x
        x_spatial = self.inception_local(x_spatial)
            
        spatial_out = self.spatial_attention(x_spatial)
        spatial_out_4d = spatial_out if is_4d else spatial_out.view(B, H, W, C)
        if z_prior is not None and lcma is not None:
            spatial_out_4d = lcma(spatial_out_4d, z_prior)

        if not is_4d:
            spatial_out = spatial_out_4d.view(B, L, C)
        else:
            spatial_out = spatial_out_4d
        
        
        if self.use_fourier:
            
            if is_4d:
                x_img = x.permute(0, 3, 1, 2).contiguous()
            else:
                x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                
            x_fre_complex = torch.fft.fft2(x_img, dim=(-2, -1), norm='ortho')
            x_fre = torch.abs(x_fre_complex) 
            
            
            x_polar, indices = polar_sorting(x_fre)
            x_polar = x_polar.permute(0, 2, 1).view(B, H, W, C) # [B, H, W, C]
            
            
            fourier_out = self.fourier_mamba(x_polar) # [B, H, W, C]
            
            
            fourier_out_flat = fourier_out.permute(0, 3, 1, 2).reshape(B, C, L)
            fourier_out_abs = polar_restore(fourier_out_flat, indices, (H, W)) # [B, C, H, W]
            
            
            x_phase = torch.angle(x_fre_complex)
            out_complex = fourier_out_abs * torch.exp(1j * x_phase)
            
            
            fourier_out_spatial_img = torch.fft.ifft2(out_complex, dim=(-2, -1), norm='ortho').real
            fourier_out_spatial = fourier_out_spatial_img.permute(0, 2, 3, 1).contiguous()
            
            
            if not is_4d:
                merged_out = self.gsff(spatial_out_4d, fourier_out_spatial)
                x_merged = input + self.drop_path(merged_out.view(B, L, C))
            else:
                x_merged = input + self.drop_path(self.gsff(spatial_out_4d, fourier_out_spatial))
        else:
            x_merged = input + self.drop_path(spatial_out)
        
        
        x_norm = self.ln_2(x_merged)
        if is_4d:
            x_img_lem = x_norm.permute(0, 3, 1, 2).contiguous()
        else:
            x_img_lem = x_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            
        x_lem = self.lem(x_img_lem).permute(0, 2, 3, 1)
        if not is_4d:
            x_lem = x_lem.reshape(B, L, C)
        
        return x_merged + self.drop_path(x_lem)


def ifft2c(x, dim=((-2,-1)), img_shape=None):
    """Documentation omitted for release."""
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def fft2c(x, dim=((-2,-1)), img_shape=None):
    """Documentation omitted for release."""
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

class DataConsistency(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, num_of_feat_maps, patchify, patch_size=2):
        super(DataConsistency, self).__init__()
        self.unpatchify = Unpatchify(num_of_feat_maps, dim_scale=patch_size)
        self.activation = nn.SiLU()
        self.patchify = patchify
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=2, embed_dim=num_of_feat_maps, norm_layer=nn.LayerNorm)

    def data_cons_layer(self, im, mask, zero_fill, coil_map):
        """Documentation omitted for release."""
        
        im_complex = im[:,0,:,:] + 1j * im[:,1,:,:]
        zero_fill_complex = zero_fill[:,0,:,:] + 1j * zero_fill[:,1,:,:]
        
        zero_fill_complex_coil_sep = torch.tile(zero_fill_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        im_complex_coil_sep = torch.tile(im_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        
        actual_kspace = fft2c(zero_fill_complex_coil_sep)
        gen_kspace = fft2c(im_complex_coil_sep)
        
        mask_bool = mask>0
        mask_coil_sep = torch.tile(mask_bool, dims=[1,coil_map.shape[1],1,1])
        gen_kspace_dc = torch.where(mask_coil_sep, actual_kspace, gen_kspace)
        
        gen_im = torch.sum(ifft2c(gen_kspace_dc) * torch.conj(coil_map), dim=1)
        
        gen_im_return = torch.stack([torch.real(gen_im), torch.imag(gen_im)], dim=1)
        return gen_im_return.type(im.dtype)
    
    def forward(self, x, zero_fill, mask, coil_map):
        """Documentation omitted for release."""
        
        h = self.unpatchify(x)  
        
        h = self.data_cons_layer(h, mask, zero_fill, coil_map)
        if self.patchify:
            
            h = self.activation(h)
            h = self.patch_embed(h)
            return x + h
        else:
            
            return h


try:
    from taming.models.vqgan import VQModel
except ImportError:
    VQModel = None

from networks.vqgan_mri import VQGAN_MRI

class FrozenPriorEncoder(nn.Module):
    def __init__(self, vqgan_ckpt_path):
        super().__init__()
        
        self.dummy = False
        
        try:
            
            self.model = VQGAN_MRI(in_channels=2, hidden_dims=[64, 128, 256, 512], z_channels=256, n_embed=1024, n_res_layers=2)
            
            
            
            ckpt = torch.load(vqgan_ckpt_path, map_location='cpu')
            if 'vqgan' in ckpt:
                self.model.load_state_dict(ckpt['vqgan'])
            else:
                
                self.model.load_state_dict(ckpt)
                
            print(f"Successfully loaded custom VQGAN from {vqgan_ckpt_path}")
            
        except Exception as e:
            print(f"Failed to load custom VQGAN: {e}")
            
            if VQModel is not None:
                try:
                    print("Attempting to load as taming-transformers VQModel...")
                    self.model = VQModel.load_from_checkpoint(vqgan_ckpt_path)
                except Exception as e2:
                    print(f"Also failed to load as taming VQModel: {e2}")
                    self.dummy = True
            else:
                self.dummy = True
        
        if not self.dummy:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: Zero-filled Image [B, 2, H, W] (Real/Imag) usually
        if self.dummy:
            return None
            
        
        
        
        
        
        
        
        
        z = self.model.encoder(x) 
        return z # [B, 256, H/4, W/4]


class FrozenCLIPPriorEncoder(nn.Module):
    """Frozen CLIP image encoder for global visual-semantic priors."""

    def __init__(self, model_name='ViT-B/32', pretrained='', download_root='', image_size=224):
        super().__init__()
        self.dummy = False
        self.backend = None
        self.image_size = image_size

        try:
            import open_clip

            open_clip_name = model_name.replace('/', '-')
            pretrained_arg = pretrained if pretrained else 'openai'
            self.model, _, _ = open_clip.create_model_and_transforms(
                open_clip_name,
                pretrained=pretrained_arg,
            )
            self.backend = 'open_clip'
            print(f"Successfully loaded frozen open_clip image encoder: {open_clip_name}, pretrained={pretrained_arg}")
        except Exception as e_open_clip:
            try:
                import clip

                kwargs = {}
                if download_root:
                    kwargs['download_root'] = download_root
                self.model, _ = clip.load(model_name, device='cpu', **kwargs)
                self.backend = 'clip'
                print(f"Successfully loaded frozen CLIP image encoder: {model_name}")
            except Exception as e_clip:
                print(f"Failed to load CLIP prior encoder. open_clip error: {e_open_clip}; clip error: {e_clip}")
                self.model = None
                self.dummy = True

        if not self.dummy:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer('clip_mean', mean, persistent=False)
        self.register_buffer('clip_std', std, persistent=False)

    def _prepare_image(self, x):
        if x.shape[1] >= 2:
            x = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-8)
        else:
            x = x[:, 0:1]

        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        x = (x - self.clip_mean.to(dtype=x.dtype, device=x.device)) / self.clip_std.to(dtype=x.dtype, device=x.device)
        return x

    def forward(self, x):
        if self.dummy:
            return None

        clip_input = self._prepare_image(x)
        model_dtype = next(self.model.parameters()).dtype
        clip_input = clip_input.to(dtype=model_dtype)

        if hasattr(self.model, 'encode_image'):
            prior = self.model.encode_image(clip_input)
        else:
            prior = self.model.visual(clip_input)
        prior = prior.float()
        prior = prior / (prior.norm(dim=-1, keepdim=True) + 1e-6)
        return prior


class LCMA(nn.Module):
    """Latent Cross-Modulation Attention from the PGD-Mamba paper."""

    def __init__(self, dim, semantic_dim, num_heads=4):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(semantic_dim, dim)
        self.v_proj = nn.Linear(semantic_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _pool_prior(self, prior):
        if prior.dim() == 2:
            return prior.unsqueeze(1)
        if prior.dim() == 4:
            return prior.flatten(2).transpose(1, 2)
        return prior

    def forward(self, x, semantic_prior):
        B, H, W, C = x.shape
        tokens = x.view(B, H * W, C)
        prior_tokens = self._pool_prior(semantic_prior)

        q = self.q_proj(tokens).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(prior_tokens).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(prior_tokens).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        prior_msg = (attn @ v).transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.norm(tokens + self.out_proj(prior_msg))
        return out.view(B, H, W, C)


class LGMM(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        
        self.modulation_net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(latent_dim, dim * 2, kernel_size=3, padding=1) 
        )
        
        
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(latent_dim + dim, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
        
        nn.init.zeros_(self.modulation_net[-1].weight)
        nn.init.zeros_(self.modulation_net[-1].bias)
        
        
        
        nn.init.constant_(self.gate_net[-2].bias, -2.0)

    def forward(self, x, z_prior):
        """
        x: Image Features [B, H, W, C]
        z_prior: Latent Code [B, C_z, H_z, W_z]
        """
        B, H, W, C = x.shape
        
        
        z_resized = F.interpolate(z_prior, size=(H, W), mode='bilinear', align_corners=False)
        
        
        style = self.modulation_net(z_resized)
        style = style.permute(0, 2, 3, 1) # [B, H, W, 2*C]
        gamma, beta = style.chunk(2, dim=-1)
        
        
        
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        gate_input = torch.cat([x_nchw, z_resized], dim=1)
        gate = self.gate_net(gate_input)
        gate = gate.permute(0, 2, 3, 1) # [B, H, W, C]
        
        
        x_norm = self.norm(x)
        modulated = x_norm * (1 + gamma) + beta
        
        
        
        out = x + gate * (modulated - x)
        
        return out

class DPH_Block(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0., d_state=16, latent_dim=256, **kwargs):
        super().__init__()
        
        self.lgmm = LGMM(hidden_dim, latent_dim) 
        self.vssm = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            window_size=kwargs.get('window_size', 0)
        )

    def forward(self, x_img, z_prior=None):
        
        if z_prior is not None:
            
            x_img_modulated = self.lgmm(x_img, z_prior)
            
            feat_img = self.vssm(x_img_modulated)
        else:
            
            feat_img = self.vssm(x_img)
        
        return feat_img


class PriorDPHBlock(nn.Module):
    """Prior-guided Mamba block supporting spatial VQGAN priors and global CLIP priors."""

    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm,
                 attn_drop_rate=0., d_state=16, latent_dim=256, **kwargs):
        super().__init__()
        self.lgmm = LGMM(hidden_dim, latent_dim)
        self.lcma = LCMA(hidden_dim, latent_dim)
        self.vssm = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            window_size=kwargs.get('window_size', 0),
            use_fourier=kwargs.get('use_fourier', False),
        )

    def forward(self, x_img, z_prior=None):
        if z_prior is not None:
            if z_prior.dim() == 2:
                return self.vssm(x_img, z_prior=z_prior, lcma=self.lcma)
            else:
                x_img = self.lgmm(x_img, z_prior)
        return self.vssm(x_img)

class VSSLayer(nn.Module):
    """Documentation omitted for release."""

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        use_prior=False,
        latent_dim=256,
        use_fourier=False, 
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.use_prior = use_prior

        
        if use_prior:
            self.blocks = nn.ModuleList([
                PriorDPHBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    latent_dim=latent_dim,
                    window_size=kwargs.get('window_size', 0),
                    use_fourier=use_fourier 
                )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    window_size=kwargs.get('window_size', 0),
                    use_fourier=use_fourier 
                )
                for i in range(depth)])
        
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x, us_im=None, us_mask=None, coil_maps=None, z_prior=None):
        """Documentation omitted for release."""
        
        for blk in self.blocks:
            if self.use_checkpoint:
                
                if self.use_prior and z_prior is not None:
                     x = checkpoint.checkpoint(blk, x, z_prior)
                else:
                     x = checkpoint.checkpoint(blk, x)
            else:
                if self.use_prior and z_prior is not None:
                    x = blk(x, z_prior)
                else:
                    x = blk(x)
        
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class VSSLayer_up(nn.Module):
    """Documentation omitted for release."""

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        window_size=0,
        use_fourier=False, 
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                window_size=window_size,
                use_fourier=use_fourier, 
                **kwargs,
            )
            for i in range(depth)])
        
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        
        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None


    def forward(self, x):
        """Documentation omitted for release."""
        
        for blk in self.blocks:
            if self.use_checkpoint:
                
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        
        if self.upsample is not None:
            x = self.upsample(x)

        return x

class VSSM_unrolled(nn.Module): 
    """Documentation omitted for release."""
    def __init__(self, patch_size=4, in_chans=2, num_classes=2, 
                 dims=[128, 128, 128, 128], depths=[2, 2, 2, 2, 2, 2],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, vqgan_ckpt_path=None, latent_dim=256,
                 use_clip_prior=False, clip_model_name='ViT-B/32', clip_pretrained='',
                 clip_download_root='', clip_image_size=224, clip_embed_dim=512,
                 window_size=0, use_fourier=False, **kwargs): 
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.use_fourier_master = use_fourier
        self.num_layers = len(depths)
        self.dims = dims
        self.patch_size = patch_size
        self.embed_dim = dims[0]
        
        
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        prior_dim = clip_embed_dim if use_clip_prior else latent_dim
        if use_clip_prior:
            self.prior_encoder = FrozenCLIPPriorEncoder(
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                download_root=clip_download_root,
                image_size=clip_image_size,
            )
            self.use_prior = not getattr(self.prior_encoder, 'dummy', False)
        elif vqgan_ckpt_path is not None and len(vqgan_ckpt_path) > 10 and "your/vqgan" not in vqgan_ckpt_path:
            self.prior_encoder = FrozenPriorEncoder(vqgan_ckpt_path)
            
            self.use_prior = not getattr(self.prior_encoder, 'dummy', False)
        else:
            self.prior_encoder = None
            self.use_prior = False

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.layers = nn.ModuleList()
        
        for i_stage in range(self.num_layers):
            stage_dim = dims[i_stage] if i_stage < len(dims) else dims[-1]
            
            
            
            use_fourier_in_stage = (True if i_stage >= 2 else False) if self.use_fourier_master else False
            use_prior_in_stage = self.use_prior and i_stage >= 2

            layer = VSSLayer(
                dim=stage_dim,
                depth=depths[i_stage],
                d_state=d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])] if isinstance(dpr, list) else 0.1,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_prior=use_prior_in_stage,
                latent_dim=prior_dim,
                window_size=self.window_size,
                use_fourier=use_fourier_in_stage 
            )
            self.layers.append(layer)
            
            self.layers.append(DataConsistency(stage_dim, patchify=True, patch_size=patch_size))

        
        self.last_dc = DataConsistency(self.embed_dim, patchify=False, patch_size=patch_size)
        
        
        self.norm = norm_layer(self.embed_dim)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, us_im, us_mask, coil_map):
        x = self.patch_embed(x)
        
        
        z_prior = None
        if self.use_prior and self.prior_encoder is not None:
            with torch.no_grad():
                z_prior = self.prior_encoder(us_im)
        
        for layer in self.layers:
            # Check if layer is VSSLayer (which has use_prior) or DataConsistency
            if isinstance(layer, VSSLayer):
                if self.use_prior:
                    x = layer(x, us_im, us_mask, coil_map, z_prior=z_prior)
                else:
                    x = layer(x, us_im, us_mask, coil_map)
            else:
                # DataConsistency layer
                x = layer(x, us_im, us_mask, coil_map)
                
        x = self.norm(x)
        return x

    def forward(self, x, us_mask, coil_map):
        us_im = x.clone()
        x = self.forward_features(x, us_im, us_mask, coil_map)
        x = self.last_dc(x, us_im, us_mask, coil_map)
        return x

    def flops(self, shape=(2, 256, 256)):
        supported_ops={
            "aten::silu": None, 
            "aten::neg": None, 
            "aten::exp": None, 
            "aten::flip": None, 
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, 
        }

        model = copy.deepcopy(self)
        try:
           model.cuda().eval()
           input1 = torch.randn((1, 2, 256, 256), device=next(model.parameters()).device)
           input2 = torch.randn((1, 5, 256, 256), device=next(model.parameters()).device)
           input3 = torch.randn((1, 1, 256, 256), device=next(model.parameters()).device)
           params = parameter_count(model)[""]
           Gflops, unsupported = flop_count(model=model, inputs=(input1, input2, input3), supported_ops=supported_ops)
           del model, input1, input2, input3
           return f"params {params} GFLOPs {sum(Gflops.values())}"
        except:
           return "FLOPs calc failed"




class VSSM(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, patch_size=4, in_chans=2, num_classes=2, depths=[2, 2, 2, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", 
                 window_size=0, use_fourier=False, **kwargs): # Added use_fourier
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.use_fourier_master = use_fourier
        self.num_layers = len(depths)
        
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample
        self.patch_size = patch_size

        
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            use_fourier_in_layer = (True if i_layer >= 2 else False) if self.use_fourier_master else False
            
            layer = VSSLayer(
                dim = int(dims[0] * 2 ** i_layer),  
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,  
                use_checkpoint=use_checkpoint,
                window_size=self.window_size,
                use_fourier=use_fourier_in_layer 
            )
            self.layers.append(layer)

        
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            concat_linear = nn.Linear(2*int(dims[0]*2**(self.num_layers-1-i_layer)),
            int(dims[0]*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            
            use_fourier_in_layer = (True if (self.num_layers - 1 - i_layer) >= 2 else False) if self.use_fourier_master else False

            if i_layer ==0 :
                
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                
                layer_up = VSSLayer_up(
                    dim= int(dims[0] * 2 ** (self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,  
                    use_checkpoint=use_checkpoint,
                    window_size=self.window_size,
                    use_fourier=use_fourier_in_layer 
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=patch_size,dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        
        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """Documentation omitted for release."""
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Documentation omitted for release."""
        
        x = self.patch_embed(x)

        
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        
        x = self.norm(x)  # B H W C
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        """Documentation omitted for release."""
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                
                x = layer_up(x)
            else:
                
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)  
                x = layer_up(x)

        
        x = self.norm_up(x)  # B H W C
  
        return x
    
    def up_x4(self, x, patch_size):
        """Documentation omitted for release."""
        if self.final_upsample=="expand_first":
            B,H,W,C = x.shape
            
            x = self.up(x)
            x = x.view(B, patch_size*H, patch_size*W, -1)
            
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            
            x = self.output(x)
            
        return x
    
    def forward(self, x):
        """Documentation omitted for release."""
        
        x,x_downsample = self.forward_features(x)
        
        x = self.forward_up_features(x,x_downsample)
        
        x = self.up_x4(x, self.patch_size)
        return x


    def flops(self, shape=(3, 224, 224)):
        """Documentation omitted for release."""
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, 
            "aten::neg": None, 
            "aten::exp": None, 
            "aten::flip": None, 
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, 
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        
        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"


# APIs with VMamba2Dp =================
def check_vssm_equals_vmambadp():
    """Documentation omitted for release."""
    from bak.vmamba_bak1 import VMamba2Dp

    
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0
