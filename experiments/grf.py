from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as topt
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil

import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from data.grf.grf import gen_grfs


# torch distribs
import torch.distributions as dists

from dawgz import job, after, ensure, schedule

import wandb

DATA_RES = 128

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=1, output_dim=1):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activ = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(DATA_RES, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, 128),
            self.activ,
            nn.Linear(128, 2 * self.latent_dim),
        )


    def forward(self, u):
        if len(u.shape) == 3 and u.shape[1] == 1:
            u = u.squeeze(1)
        out = self.mlp(u)
        mean = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=1, output_dim=1):
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

class INRDecoder(Decoder):
    def __init__(
            self,
            latent_dim=64,
            input_dim=1,
            output_dim=1,
            pe_var=10.0,
            use_pe=True,
            pe_m='half_ldim',
            pe_interleave=True,
            device='cpu'
    ):
        super().__init__(latent_dim, input_dim, output_dim)

        self.activ = nn.GELU()

        self.use_pe = use_pe
        self.pe_interleave = pe_interleave
        self.pe_var = pe_var
        if pe_m == 'half_ldim':
            self.m = self.latent_dim // 2
        else:
            self.m = pe_m
        self.pe_dist = dists.Normal(0.0, self.pe_var)
        self.B = self.pe_dist.sample((self.m, input_dim)).to(device)

        # (original) NeRF-like architecture
        if use_pe:
            self.mlp_x = nn.Sequential(
                nn.Linear(self.latent_dim, 128),  # With positional encoding
                self.activ,
            )
        else:
            self.mlp_x = nn.Sequential(
                nn.Linear(self.input_dim, 128),  # No positional encoding
                self.activ,
            )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            self.activ,
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            self.activ,
            nn.Linear(in_features=128, out_features=output_dim),
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
    def __init__(self, latent_dim=64, input_dim=1, *args, **kwargs):
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

DECODERS = {
    "inr": INRDecoder,
    "linear": LinearDecoder,
}

class VANO(nn.Module):
    def __init__(self,
                 latent_dim=64,
                 input_dim=1,
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

        ls = torch.linspace(0, 1, DATA_RES).to(device)
        self.grid = torch.stack(torch.meshgrid(ls), dim=-1).unsqueeze(0)

    def forward(self, u, custom_grid=None):
        """
        sample: use p(z) instead of p(z | u)
        custom_grid: use custom res (super or sub resolution)
        """

        mean, logvar = self.encoder(u)

        eps = torch.randn(u.shape[0], self.latent_dim, device=u.device)
        z = mean + eps * torch.exp(0.5 * logvar)

        grid = self.grid if custom_grid is None else custom_grid
        grids = grid.expand(u.shape[0], *grid.shape[1:])
        u_pred = self.decoder(grids, z)

        return mean, logvar, z, u_pred

    def sample(self, device, z=None, n_samples=1, custom_grid=None):
        """
        sample: use p(z) instead of p(z | u)
        custom_grid: use custom res (super or sub resolution)
        """
        if z is None:
            z = torch.randn(n_samples, self.latent_dim, device=device)
        grid = self.grid if custom_grid is None else custom_grid
        grids = grid.expand(n_samples, *grid.shape[1:])
        u_pred = self.decoder(grids, z)

        return u_pred


def load_data(N=1, device='cpu'):
    grid, u = gen_grfs(N)

    return grid.to(device), u.to(device)

def is_slurm():
    return shutil.which('sbatch') is not None

@job(
    array=1,
    partition="a5000,tesla,quadro,2080ti",
    cpus=4,
    gpus=1,
    ram="16GB",
    time="24:00:00",
    name="vanogrf",
)
def train(i: int):
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Data
    N_train = 2048
    train_data = load_data(N_train, device=device)
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    N_test = 128
    test_data = load_data(N_test, device=device)
    test_dataset = torch.utils.data.TensorDataset(*test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Training
    decoder = 'linear'
    vano = VANO(
        latent_dim=64,
        decoder=decoder,
        decoder_args=dict(
            use_pe=True,
            pe_m='half_ldim',
        ),
        device=device
    ).to(device)
    vano.train()

    # Parameters:
    S = 4  # Monte Carlo samples for evaluating reconstruction loss in ELBO (E_q(z | x) [log p(x | z)])
    #beta = 1e-5  # Weighting of KL divergence in ELBO
    beta = 1e-3  # β
    recon_reduction = 'mean'  # Reduction of reconstruction loss over grid points (mean or sum)
    batch_size = 32
    num_iters = 25_000

    # Exponential decay of every 1000 iterations by 0.9
    lr = 1e-3
    lr_decay = 0.9
    lr_decay_every = 1000
    optimizer = topt.Adam(vano.parameters(), lr=lr)
    lr_scheduler = topt.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)

    # paper: Approximation of gaussian error in Banach spaces
    # mse: Typical mean squared error, as for finite data
    # ce: Cross-entropy loss, as for finite data (with Bernoulli assumption)
    recon_loss = 'paper'

    # W&B
    wandb_enabled = is_slurm()
    if wandb_enabled:
        wandb.init(
            project="vano",
            entity='slewin',
            name=f"grf",
            config={
                "S": S,
                "beta": beta,
                "recon_loss": recon_loss,
                "recon_reduction": recon_reduction,
                "batch_size": batch_size,
                "num_iters": num_iters,
                "n_train": N_train,
                "n_test": N_test,
                "lr": lr,
                "latent-dim": vano.latent_dim,
                "lr_decay": lr_decay,
                "lr_decay_every": lr_decay_every,
                "experiment-name": "grf",
            }
        )

    step = 0
    num_epochs = num_iters // len(train_loader)
    #num_epochs = max(num_epochs, 10)  # For experiment on N_train.
    for epoch in range(num_epochs):
        for grid, u in train_loader:
            mu, logvar, z, u_hat = vano(u.view(-1, DATA_RES))
            u_hat = u_hat.squeeze()
            u, u_hat = u.flatten(1), u_hat.flatten(1)
            # Sample S values of z
            eps = torch.randn(S, *z.shape, device=z.device)
            z_samples = mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)
            z_samples = z_samples.view(S * batch_size, *z_samples.shape[2:])
            u_hat_samples = vano.decoder(grid[:1].expand(z_samples.shape[0], *grid.shape[1:]).unsqueeze(-1), z_samples)
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
                "kl_loss_scaled": (beta * kl_loss).item(),
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }

            if step % 100 == 0:
                with torch.no_grad():
                    vano.eval()

                    # ----- Reconstruction Image -----
                    test_u = test_dataset[0][1]
                    test_u_hat = vano(test_u.view(-1, 1, DATA_RES))[3].squeeze()

                    fig, ax = plt.subplots()
                    plt.plot(grid[0].cpu(), test_u.cpu(), label='u')
                    plt.plot(grid[0].cpu(), test_u_hat.cpu(), label='u_hat')
                    ax.legend()
                    log_dict["reconstr_img"] = fig

                    """
                    # ----- Latent walk -----
                    u_start = test_dataset[0][1]
                    u_end = test_dataset[torch.randint(0, len(test_dataset), (1,))][1].squeeze()
                    us = torch.stack([u_start, u_end]).view(-1, 1, DATA_RES)
                    zs = vano.encoder(us)[0]
                    z_start = zs[0]
                    z_end = zs[1]
                    z_walk = torch.stack([z_start + (z_end - z_start) * (i / 10) for i in range(10)], dim=0)
                    grids = vano.grid.expand(10, *vano.grid.shape[1:])
                    test_u_walk = vano.decoder(grids, z_walk).squeeze()  # [10, 48, 48]
                    u_null = torch.zeros(DATA_RES, DATA_RES).to(device)
                    first_row = torch.cat([u_start] + 8 * [u_null] + [u_end], dim=1) # [48, 480]
                    # Display latent walk u's next to each other
                    second_row = torch.cat(list(test_u_walk), dim=1)  # [48, 480]
                    latent_walk = torch.cat([first_row, second_row]).detach().cpu().numpy()
                    latent_walk = plt.get_cmap('viridis')(latent_walk)[:, :, :3]
                    log_dict["latent_walk"] = wandb.Image(latent_walk)
                    """

                    # ----- Random sampling -----
                    img_grid_size = 10
                    # P(z) = N(0, I)
                    z = torch.randn(img_grid_size**2, vano.latent_dim).to(device)
                    grids = vano.grid.expand(img_grid_size**2, *vano.grid.shape[1:])
                    us = vano.decoder(grids, z).squeeze()
                    us = us.view(img_grid_size, img_grid_size, DATA_RES)
                    fig, ax = plt.subplots()
                    plt.plot(grid[0].cpu(), us[0, 0].cpu(), label='u')
                    log_dict["rand_samples"] = fig

                    vano.train()

                    """
                    # ----- Super Res ----
                    resolutions = [
                        4, 8, 16, 28, 64, 128, 256
                    ]
                    max_res = max(resolutions)
                    upsample = lambda x: F.interpolate(x, size=max_res, mode='nearest').squeeze()
                    empty = torch.zeros(max_res).detach().cpu()
                    test_u = test_dataset[0][1][None, None, ...]
                    multires_samples = [empty]
                    multires_decoded = [upsample(test_u).squeeze().detach().cpu()]
                    z = torch.randn(1, vano.latent_dim).to(device)
                    for res in resolutions:
                        ls = torch.linspace(0, 1, res).to(device)
                        grid = torch.stack(torch.meshgrid(ls), dim=-1).unsqueeze(0)
                        test_u_hat = vano(test_u, custom_grid=grid)[3]
                        sample_u_hat = vano.sample(device, z=z, custom_grid=grid)

                        test_u_hat = test_u_hat.permute(0, 3, 1, 2)
                        sample_u_hat = sample_u_hat.permute(0, 3, 1, 2)

                        # Upsample with no interpolation to max_res
                        test_u_hat = upsample(test_u_hat)
                        sample_u_hat = upsample(sample_u_hat)

                        multires_samples.append(sample_u_hat.detach().cpu())
                        multires_decoded.append(test_u_hat.detach().cpu())

                    multires_samples = torch.cat(multires_samples, dim=1)
                    multires_decoded = torch.cat(multires_decoded, dim=1)

                    whole_grid = torch.cat([multires_samples, multires_decoded], dim=0).numpy()
                    whole_grid = plt.get_cmap('viridis')(whole_grid)[:, :, :3]

                    log_dict["multires"] = wandb.Image(whole_grid)
                    """


            if wandb_enabled:
                wandb.log(log_dict, step=step)

            step += 1

    # Save model
    torch.save(vano.state_dict(), "vano.pt")

if __name__ == "__main__":
    # Check if srun command exists in os
    backend = "slurm" if is_slurm() else "async"
    # Check if argument "--local" is given then choose async backend
    if "--local" in sys.argv:
        backend = "async"
    print(f"Launching {len(train.array)} jobs with backend {backend}")
    schedule(
        train,
        backend=backend,
        export="ALL",
        shell="/bin/sh",
        env=["export WANDB_SILENT=true"],
    )