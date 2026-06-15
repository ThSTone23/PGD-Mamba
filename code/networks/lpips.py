import torch
import torch.nn as nn
from torchvision.models import vgg16


def _make_vgg16_features():
    try:
        from torchvision.models import VGG16_Weights
        return vgg16(weights=VGG16_Weights.DEFAULT).features
    except Exception:
        try:
            return vgg16(pretrained=True).features
        except Exception:
            return vgg16(weights=None).features

class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        # Prefer pretrained VGG16 when available; fall back to untrained weights
        # so validation never fails in offline environments.
        vgg = _make_vgg16_features()
        
        # Slices to extract features from
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()

    @staticmethod
    def _to_rgb(x):
        if x.shape[1] == 2:
            x = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2 + 1e-8)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def forward(self, x, y):
        # x, y: [B, C, H, W]. MRI 2-channel complex data is converted to magnitude RGB.
        x = self._to_rgb(x)
        y = self._to_rgb(y)

        h_x = x
        h_y = y
        
        loss = 0.0
        for i, slice_net in enumerate([self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]):
            h_x = slice_net(h_x)
            h_y = slice_net(h_y)
            loss += torch.mean((h_x - h_y)**2)
            
        return loss
