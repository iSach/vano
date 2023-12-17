import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as topt
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import shutil

# torch distribs
import torch.distributions as dists

from dawgz import job, after, ensure, schedule

import wandb

# TODO:
# - Linear Decoder: sum z_i * MLP(x_i)
# - Concat at each layer Decoder (Attention beats concatenation for conditioning neural fields) (Left of Fig. 4)
# - (Periodic and other) Positional encodings (VANO)
# - Hyper-network decoder (Attention beats concatenation for conditioning neural fields) (Middle of Fig. 4)
#       -- This is not tried in VANO
#       -- z -> MLP -> [weights for MLP(x)]
# - Attention Decoder (Attention beats concatenation for conditioning neural fields) (Right of Fig. 4)
#       -- This is not tried in VANO

# Make decoders etc a common thing between experiments, with custom activations, nb of layers etc
# (ie cleaner overall lol)

# TODO:
# - Multi-scale
# - Super-resolution

# NB:
# - VAE uses multi-head attention between z [Hz x Wz x Dz] and (x, d)
#   each attention block receives a different channel (Dz) 

class Encoder(nn.Module):
    def __init__(self, latent_dim=256, input_dim=2, output_dim=1):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activ = nn.GELU()

        # Input: [output_dim, 128, 128]
        # "In all other benchmarks we build encoders using 
        #   a simple VGG-style deep convolutional network 
        #   [Simonyan & Zisserman 2014], where in each layer 
        #   the input feature maps are down-sampled by a factor 
        #   of 2 using strided convolutions, while the number 
        #   of channels are doubled."
        # Note: this is incorrect wrt VGG. VGG uses 1-stride convolutions
        #       with max pooling (2x2 with stride 2) every 2 convolutions.
        self.seq = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=2),  # [8, 64, 64]
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2),  # [16, 32, 32]
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),  # [32, 16, 16]
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),  # [64, 8, 8]
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),  # [128, 4, 4]
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),  # [256, 2, 2]
            nn.GELU(),
            nn.Flatten(),  # [1024]
            nn.Linear(1024, latent_dim * 2),
        )

    def forward(self, u):
        out = self.seq(u)
        mean = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]

        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, input_dim=2, output_dim=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _expand_z(self, x, z):
        """
        Expands z to match x's shape.

        Batches are often of shape [batch_size, grid_H, grid_W, 2]
        while there is one z per batch element (i.e., [batch_size, latent_dim])
        Expanding allows to more easily process between x and z.

        Args:
            x: (batch_size, input_dim) tensor of spatial locations
            z: (batch_size, latent_dim) tensor of latent representations
        """
        _, z_dim = z.shape
        z = z.view(-1, *([1] * (len(x.shape) - 2)), z_dim)
        z = z.expand(-1, *x.shape[1:-1], z_dim)
        return z
    
    def forward(self, x, z):
        """
        Computes u(x) by conditioning on z, a latent representation of u.

        Parameters
        ----------
        x : [batch_size, ..., in_dim] tensor of spatial locations
        z : [batch_size, latent_dim] tensor of latent representations
        """
        return self.decode(x, self._expand_z(x, z))

    # This function is likely to change with a proper
    # implementation of Pos. Encodings
    # TODO: think about a clean way to handle
    # the shape of x's in practice.
    def decode(self, x, z):
        """
        Computes u(x) by conditioning on z, a latent representation of u.

        This function takes as input expanded z's 
        i.e., z's shape is the same as x's shape.

        Parameters
        ----------
        x : [batch_size, ..., in_dim] tensor of spatial locations
        z : [batch_size, ..., latent_dim] tensor of latent representations
        """
        raise NotImplementedError
    
# TODO make positional encoding separate
# TODO try with/without positional encoding
# Seed? For random gaussian fourier features positional encoding

# NeRF-like decoder
class NeRFDecoder(Decoder):
    def __init__(
            self, 
            latent_dim=32, 
            input_dim=2, 
            output_dim=1, 
            pe_var=10.0, 
            use_pe=True, 
            pe_interleave=True,
            device='cpu'
    ):
        super().__init__(latent_dim, input_dim, output_dim)

        self.activ = nn.GELU()

        self.use_pe = use_pe
        self.pe_interleave = pe_interleave
        self.pe_var = pe_var
        self.m = self.latent_dim // 2
        self.pe_dist = dists.Normal(0.0, self.pe_var)
        self.B = self.pe_dist.sample((self.m, input_dim)).to(device)

        # (original) NeRF-like architecture
        if use_pe:
            self.mlp_x = nn.Sequential(
                #nn.Linear(self.input_dim, 256),  # No positional encoding
                nn.Linear(self.latent_dim, 256),  # With positional encoding
                self.activ,
                nn.Linear(256, 256),
                self.activ,
                nn.Linear(256, 256),
                self.activ,
                nn.Linear(256, 128)
            )
        else:
            self.mlp_x = nn.Sequential(
                nn.Linear(self.input_dim, 256),  # No positional encoding
                self.activ,
                nn.Linear(256, 256),
                self.activ,
                nn.Linear(256, 256),
                self.activ,
                nn.Linear(256, 128)
            )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * self.latent_dim),
            self.activ,
            nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            self.activ,
            nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            self.activ,
            nn.Linear(4 * self.latent_dim, 128)
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(256, 256),
            self.activ,
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        """
        Computes u(x) by conditioning on z, a latent representation of u.

        Args:
            x: (batch_size, input_dim) tensor of spatial locations
            z: (batch_size, latent_dim) tensor of latent representations
        """

        if self.use_pe:
            v = torch.einsum('ij, ...j -> ...i', self.B, x)
            cos_v = torch.cos(2 * torch.pi * v)
            sin_v = torch.sin(2 * torch.pi * v)
            if self.pe_interleave:
                # [cos, sin, cos, sin, cos, sin...]
                v = torch.stack([cos_v, sin_v], dim=-1).flatten(-2, -1)
            else:
                # [cos, cos, cos, ..., sin, sin, sin, ...]
                v = torch.cat([cos_v, sin_v], dim=-1)
        else:
            v = x

        v = self.mlp_x(v)
        # Perform MLP on z before expanding to avoid
        # extra computations on the expanded z
        # Even if view/expand do not allocate more memory,
        # the operations are still performed on the expanded z.
        z = self.mlp_z(z)
        # z is [32, 32], reshape to [32, 64, 64, 32]
        z = self._expand_z(v, z)
        vz = torch.cat([v, z], dim=-1)
        vz = self.joint_mlp(vz)

        return vz
    
class LinearDecoder(Decoder):
    def __init__(self, latent_dim=32, input_dim=2, *args, **kwargs):
        super().__init__(latent_dim, input_dim, output_dim=1)

        self.activ = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, latent_dim),
        )

        self.output_activ = nn.Softplus()

    def decode(self, x, z):
        prod = self.mlp(x) * z
        dotprod = prod.sum(axis=-1)
        return self.output_activ(dotprod)

class Cat1stDecoder(Decoder):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1):
        super().__init__(latent_dim, input_dim, output_dim)

        self.activ = nn.GELU()
        self.output_activ = nn.Softplus()

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim + self.input_dim, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, output_dim),
            self.output_activ,
        )

    def decode(self, x, z):
        return self.mlp(torch.cat([x, z], dim=-1))

class DistribCatDecoder(Decoder):
    def __init__(self, latent_dim=256, input_dim=2, output_dim=1, device='cpu'):
        """
        latent_dim must be divisible by 4 (nb. hidden layers)
        """
        super().__init__(latent_dim, input_dim, output_dim)

        nb_hidden_layers = 2
        
        self.split_zdim = self.latent_dim // nb_hidden_layers

        self.nb_neurons = 64
        self.input_lin = nn.Linear(self.input_dim, self.nb_neurons)
        # 8 hidden layers
        self.hidden_lins = nn.ModuleList([
            nn.Linear(self.nb_neurons + self.split_zdim, self.nb_neurons) for _ in range(nb_hidden_layers - 1)
        ])
        self.output_lin = nn.Linear(self.nb_neurons + self.split_zdim, output_dim)
        self.activ = nn.GELU()

    def decode(self, x, z):
        zs = torch.split(z, self.split_zdim, dim=-1)  # list of 8 [..., latent_dim//8]
        x = self.input_lin(x)
        x = self.activ(x)
        for i, lin in enumerate(self.hidden_lins):
            x = torch.cat([x, zs[i]], dim=-1)
            x = lin(x)
            x = self.activ(x)
        x = torch.cat([x, zs[-1]], dim=-1)
        x = self.output_lin(x)
        return x

class HyperNetDecoder(Decoder):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1):
        super().__init__(latent_dim, input_dim, output_dim)

        self.activ = nn.GELU()

        # 33537 parameters for 3 hidden layers at 128
        # 2 * 128 + 128    = 384
        # 128 * 128 + 128  = 16512
        # 128 * 128 + 128  = 16512
        # 128 * 1 + 1      = 129
        # Total            = 33,537
        self.hyper_net = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            self.activ,
            nn.Linear(128, 256),
            self.activ,
            nn.Linear(256, 256),
            self.activ,
            nn.Linear(256, 33537),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, output_dim),
            nn.Softplus(),
        )

    def decode(self, x, z):
        W = self.hyper_net(z)

        # W is [batch_size, 33537]
        # Use W as weights for MLP
        # TODO (pay attention to batch and x's shape which will often be [batch_size, grid_H, grid_W, 2])
        pass

class AttentionDecoder(Decoder):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1):
        super().__init__(latent_dim, input_dim, output_dim)

    def forward(self, x, z):
        return x * z


DECODERS = {
    #"nerf": NeRFDecoder,
    #"linear": LinearDecoder,
    #"cat1st": Cat1stDecoder,
    "distribcat": DistribCatDecoder,
    #"hypernet": HyperNetDecoder,
    #"attention": AttentionDecoder
}

class VANO(nn.Module):
    def __init__(self, 
                 latent_dim=256, 
                 input_dim=2, 
                 output_dim=1, 
                 decoder="nerf",
                 decoder_args={},
                 device='cpu'):
        super(VANO, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
    
        self.encoder = Encoder(latent_dim, input_dim, output_dim)
        self.decoder = DECODERS[decoder](latent_dim, input_dim, output_dim, **decoder_args, device=device)

        ls = torch.linspace(0, 1, 128).to(device)
        self.grid = torch.stack(torch.meshgrid(ls, ls, indexing='ij'), dim=-1).unsqueeze(0)

    def forward(self, u):
        # Encode
        mean, logvar = self.encoder(u)
    
        # Sample
        eps = torch.randn(u.shape[0], self.latent_dim, device=u.device)
        z = mean + eps * torch.exp(0.5 * logvar)

        # Decode
        grids = self.grid.expand(u.shape[0], *self.grid.shape[1:])
        u_pred = self.decoder(grids, z)

        return mean, logvar, z, u_pred
    
# TODO multi-res hash encoding
    
import os
import glob
def load_insar(n_train=4096, res=128):
    # Get base folder of python project
    files = glob.glob(f'data/insar/data128/*.int', recursive=True)[:n_train]
    print(f"Found {len(files)} files.")
    phi_train = torch.zeros(n_train, res, res).float()
    cos_train = torch.zeros(n_train, res, res).float()
    sin_train = torch.zeros(n_train, res, res).float()
    for i, f in enumerate(files):
        dtype = np.float32
        nline = 128
        nsamp = 128

        with open(f, 'rb') as fn:
            load_arr = np.frombuffer(fn.read(), dtype=dtype)
            img = np.array(load_arr.reshape((nline, nsamp, -1)))

        phi = np.angle(img[:,:,0] + img[:,:,1]*1j)
        phi = torch.tensor(phi[:res, :res])
        
        phi_train[i] = phi
        cos_train[i] = torch.cos(phi)
        sin_train[i] = torch.sin(phi)

    return phi_train, cos_train, sin_train

def load_data(N=1, device='cpu'):

    # TODO: what is the grid for InSAR?
    
    dim_range = torch.linspace(0, 1, 128)
    grid = torch.stack(torch.meshgrid(dim_range, dim_range, indexing='ij'), dim=-1)
    grid = torch.stack([grid] * N, dim=0)

    u = load_insar(n_train=N)[0]

    return grid, u

def is_slurm():
    return shutil.which('sbatch') is not None

configs = [
    ...
]

@job(
    array=len(configs),
    partition="a5000,tesla,quadro",
    cpus=4,
    gpus=1,
    ram="16GB",
    time="24:00:00",
    name="vano",
)
def train(i: int):
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = 'cpu'

    # Data
    N_train = 4096
    train_data = load_data(N_train, device=device)
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    N_test = 128
    test_data = load_data(N_test, device=device)
    test_dataset = torch.utils.data.TensorDataset(*test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Training
    decoder = 'distribcat'
    vano = VANO(
        decoder=decoder,
        device=device
    ).to(device)
    vano.train()

    print(f"VANO has {sum(p.numel() for p in vano.parameters())} parameters.")

    # Parameters:
    S = 1  # Monte Carlo samples for evaluating reconstruction loss in ELBO (E_q(z | x) [log p(x | z)])
    #beta = 1e-5  # Weighting of KL divergence in ELBO
    beta = 1e-4
    recon_reduction = 'mean'  # Reduction of reconstruction loss over grid points (mean or sum)
    batch_size = 16
    num_iters = 25_000

    # Exponential decay of every 1000 iterations by 0.9
    lr = 1e-3
    lr_decay = 0.9
    lr_decay_every = 1000
    optimizer = topt.Adam(vano.parameters(), lr=lr)
    lr_scheduler = topt.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)

    # W&B
    wandb_enabled = is_slurm()
    if wandb_enabled:
        wandb.init(
            project="vano",
            name=f"Please work",
            config={
                "S": S,
                "beta": beta,
                "recon_reduction": recon_reduction,
                "batch_size": batch_size,
                "num_iters": num_iters,
                "n_train": N_train,
                "n_test": N_test,
                "lr": lr,
                "lr_decay": lr_decay,
                "lr_decay_every": lr_decay_every,
                "experiment-name": "InSAR",
            }
        )

    # paper: Approximation of gaussian error in Banach spaces
    # mse: Typical mean squared error, as for finite data
    # ce: Cross-entropy loss, as for finite data (with Bernoulli assumption)
    recon_loss = 'ce'

    step = 0
    num_epochs = num_iters // len(train_loader)
    #num_epochs = max(num_epochs, 10)  # For experiment on N_train.
    for epoch in range(num_epochs):
        for grid, u in train_loader:
            grid, u = grid.to(device), u.to(device)

            mu, logvar, z, u_hat = vano(u.view(-1, 1, 128, 128))
            u_hat = u_hat.squeeze()

            u, u_hat = u.flatten(1), u_hat.flatten(1)

            # Sample S values of z
            eps = torch.randn(S, *z.shape, device=z.device)
            z_samples = mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)
            z_samples = z_samples.view(S * batch_size, *z_samples.shape[2:])
            u_hat_samples = vano.decoder(grid[:1].expand(z_samples.shape[0], *grid.shape[1:]), z_samples).squeeze()
            u_hat_samples = u_hat_samples.view(S, batch_size, *u_hat_samples.shape[1:])
            # u^: Shape=[4, bs, 64, 64, 1]
            u_hat_samples = u_hat_samples.flatten(start_dim=-2)
            # u^: Shape=[4, bs, 4096]
            # u:  Shape=[bs, 4096]
            u = u.unsqueeze(0).expand(S, *u.shape)
            # u:  Shape=[4, bs, 4096]

            if recon_loss == 'paper':
                # ELBO = E_p(eps)[log p(x | z=g(eps, x))] - KL(q(z | x) || p(z))
                # ----------------------------------------------------------------
                # Reconstruction loss: E_Q(z|x)[1/2 ||D(z)||^2_L2 - <D(z), u>^~]
                # 1/2 * ||D(z)||^2_(L^2) ~= sum_{i=1}^m D(z)(x_i) * D(z)(x_i) (?)
                Dz_norm = 0.5 * (u_hat_samples * u_hat_samples)
                # <D(z), u>^~ ~= sum_{i=1}^m D(z)(x_i) * u(x_i)
                inner_prod = u_hat_samples * u
                reconstr_loss = Dz_norm - inner_prod
            elif recon_loss == 'mse':
                reconstr_loss = F.mse_loss(u_hat_samples, u, reduction='none')
            elif recon_loss == 'ce':
                reconstr_loss = F.binary_cross_entropy(u_hat_samples, u, reduction='none')

            # Reduction
            if recon_reduction == 'mean':
                reconstr_loss = reconstr_loss.mean(axis=-1)  # Mean over grid points
            elif recon_reduction == 'sum':
                reconstr_loss = reconstr_loss.sum(axis=-1)   # Sum over grid points
            reconstr_loss = reconstr_loss.mean(axis=0)  # Mean over S
            reconstr_loss = reconstr_loss.mean(axis=0)  # Mean over batch

            kl_loss = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(axis=1).mean()

            loss = reconstr_loss + beta * kl_loss
            
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            lr_scheduler.step()

            log_dict = {
                "reconstr_loss": reconstr_loss.item(),
                "kl_loss": kl_loss.item(),
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }

            if step % 100 == 0:
                with torch.no_grad():
                    vano.eval()

                    # ----- Reconstruction Image -----
                    test_u = test_dataset[0][1]
                    test_u_hat = vano(test_u.view(-1, 1, 128, 128))[3].squeeze()
                    reconstr_img = torch.cat([test_u, test_u_hat], axis=1).detach().cpu().numpy()
                    reconstr_img = plt.get_cmap('viridis')(reconstr_img)[:, :, :3]
                    log_dict["reconstr_img"] = wandb.Image(reconstr_img)

                    # ----- Latent walk -----
                    u_start = test_dataset[0][1]
                    u_end = test_dataset[torch.randint(0, len(test_dataset), (1,))][1].squeeze()
                    us = torch.stack([u_start, u_end]).view(-1, 1, 128, 128)
                    zs = vano.encoder(us)[0]
                    z_start = zs[0]
                    z_end = zs[1]
                    z_walk = torch.stack([z_start + (z_end - z_start) * (i / 10) for i in range(10)], dim=0)
                    grids = vano.grid.expand(10, *vano.grid.shape[1:])
                    test_u_walk = vano.decoder(grids, z_walk).squeeze()  # [10, 48, 48]
                    u_null = torch.zeros(128, 128).to(device)
                    first_row = torch.cat([u_start] + 8 * [u_null] + [u_end], dim=1) # [48, 480]
                    # Display latent walk u's next to each other
                    second_row = torch.cat(list(test_u_walk), dim=1)  # [48, 480]
                    latent_walk = torch.cat([first_row, second_row]).detach().cpu().numpy()
                    latent_walk = plt.get_cmap('viridis')(latent_walk)[:, :, :3]
                    log_dict["latent_walk"] = wandb.Image(latent_walk)

                    # ----- Random sampling -----
                    img_grid_size = 10
                    # P(z) = N(0, I)
                    z = torch.randn(img_grid_size**2, vano.latent_dim).to(device)
                    grids = vano.grid.expand(img_grid_size**2, *vano.grid.shape[1:])
                    us = vano.decoder(grids, z).squeeze()
                    us = us.view(img_grid_size, img_grid_size, 128, 128)
                    us = us.permute(0, 2, 1, 3).reshape(img_grid_size * 128, img_grid_size * 128)
                    rand_samples = us.detach().cpu().numpy()
                    rand_samples = plt.get_cmap('viridis')(rand_samples)[:, :, :3]
                    log_dict["rand_samples"] = wandb.Image(rand_samples)

                    vano.train()


            if wandb_enabled:
                wandb.log(log_dict, step=step)

            step += 1

    # Save model
    torch.save(vano.state_dict(), "vano.pt")

if __name__ == "__main__":
    # Check if srun command exists in os
    backend = "slurm" if is_slurm() else "async"
    print(f"Launching {len(train.array)} jobs with backend {backend}")
    schedule(
        train,
        backend=backend,
        export="ALL",
        shell="/bin/sh",
        env=["export WANDB_SILENT=true"],
    )