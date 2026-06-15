import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # z: [b, c, h, w] -> [b, h, w, c]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # reshape back to [b, c, h, w]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, min_encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.channel_up = nn.Identity()

    def forward(self, x):
        return self.channel_up(x) + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, z_channels, n_res_layers):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_dims[0], 3, 1, 1)]
        
        for i in range(len(hidden_dims)-1):
            dim = hidden_dims[i]
            next_dim = hidden_dims[i+1]
            for _ in range(n_res_layers):
                layers.append(ResidualBlock(dim, dim))
            # Downsample
            layers.append(nn.Conv2d(dim, next_dim, 4, 2, 1))
            layers.append(nn.SiLU())
            
        # Final layers
        dim = hidden_dims[-1]
        for _ in range(n_res_layers):
            layers.append(ResidualBlock(dim, dim))
            
        layers.append(nn.GroupNorm(32, dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(dim, z_channels, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dims, z_channels, n_res_layers):
        super().__init__()
        # hidden_dims expected in order of Encoder (small -> large), need to reverse
        # But commonly we pass [64, 128, 256]. Encoder: 64->128->256. Decoder: 256->128->64.
        
        hidden_dims = hidden_dims[::-1] 
        
        layers = [nn.Conv2d(z_channels, hidden_dims[0], 3, 1, 1)]
        
        for i in range(len(hidden_dims)-1):
            dim = hidden_dims[i]
            next_dim = hidden_dims[i+1]
            for _ in range(n_res_layers):
                layers.append(ResidualBlock(dim, dim))
            # Upsample
            layers.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
            layers.append(nn.Conv2d(dim, next_dim, 3, 1, 1))
            
        # Final layers
        dim = hidden_dims[-1]
        for _ in range(n_res_layers):
            layers.append(ResidualBlock(dim, dim))
            
        layers.append(nn.GroupNorm(32, dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(dim, out_channels, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class VQGAN_MRI(nn.Module):
    def __init__(self, in_channels=2, hidden_dims=[64, 128, 256], z_channels=256, n_embed=1024, n_res_layers=2):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, z_channels, n_res_layers)
        self.decoder = Decoder(in_channels, hidden_dims, z_channels, n_res_layers) # Out channels same as in
        self.quantizer = VectorQuantizer(n_embed, z_channels, beta=0.25)
        self.pre_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        z_q, loss, _ = self.quantizer(z)
        z_q_dec = self.post_quant_conv(z_q)
        x_recon = self.decoder(z_q_dec)
        return x_recon, loss
