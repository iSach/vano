import os
import shutil

import numpy as np
import torch
# torch distribs
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as topt
import torch.utils.data as tud
import torchvision as tv
import torchvision.transforms as transforms
from dawgz import job, schedule

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

HEIGHT = 218
WIDTH = 178


class Encoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activ = nn.GELU()

        self.seq = nn.Sequential(
            # [3, 218, 178]
            nn.Conv2d(output_dim, 8, kernel_size=2, stride=2, padding=1),
            # [8, 110, 90]
            self.activ,
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            # [16, 54, 44]
            self.activ,
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            # [32, 52, 42]
            self.activ,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # [64, 25, 20]
            self.activ,
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            # [128, 8, 6]
            self.activ,
            nn.Flatten(),
            nn.Linear(128 * 8 * 6, 256),
            self.activ,
            nn.Linear(256, 2 * self.latent_dim),
        )

    def forward(self, u):
        out = self.seq(u)
        mean = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
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

# TODO positional encoding causes a lot of issues with MNIST
# - Maybe due to 28x28?
# - Experiment...

# NeRF-like decoder
class NeRFDecoder(Decoder):
    def __init__(
            self,
            latent_dim=64,
            input_dim=2,
            output_dim=3,
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
                # nn.Linear(self.input_dim, 256),  # No positional encoding
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
                # nn.Linear(256, 256),
                # self.activ,
                # nn.Linear(256, 256),
                # self.activ,
                nn.Linear(256, 128)
            )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * self.latent_dim),
            # self.activ,
            # nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            # self.activ,
            # nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            # self.activ,
            nn.Linear(4 * self.latent_dim, 128)
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(256, 256),
            self.activ,
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

        # print(f"v.shape: {v.shape}")  # [B, 218, 178, 2]
        v = self.mlp_x(v)
        # print(f"v.shape: {v.shape}")  # [B, 218, 178, 128]
        # Perform MLP on z before expanding to avoid
        # extra computations on the expanded z
        # Even if view/expand do not allocate more memory,
        # the operations are still performed on the expanded z.
        z = self.mlp_z(z)
        # print(f"z.shape: {z.shape}")  # [B, 128]
        # z is [32, 32], reshape to [32, 64, 64, 32]
        z = self._expand_z(v, z)
        # print(f"z.shape: {z.shape}")  # [B, 218, 178, 128]
        vz = torch.cat([v, z], dim=-1)
        # print(f"vz.shape: {vz.shape}")  # [16, 218, 178, 256]
        vz = self.joint_mlp(vz)
        # print(f"vz.shape: {vz.shape}")  # [16, 218, 178, 3]

        return vz


class LinearDecoder(Decoder):
    def __init__(self, latent_dim=64, input_dim=2, *args, **kwargs):
        super().__init__(latent_dim, input_dim, output_dim=3)

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
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
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
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
        """
        latent_dim must be divisible by 4 (nb. hidden layers)
        """
        super().__init__(latent_dim, input_dim, output_dim)

        self.split_zdim = self.latent_dim // 4

        self.lin1 = nn.Linear(self.input_dim, 128)
        self.lin2 = nn.Linear(128 + self.split_zdim, 128)
        self.lin3 = nn.Linear(128 + self.split_zdim, 128)
        self.lin4 = nn.Linear(128 + self.split_zdim, 128)
        self.lin5 = nn.Linear(128 + self.split_zdim, output_dim)
        self.activ = nn.GELU()
        self.output_activ = nn.Softplus()

    def decode(self, x, z):
        # TODO cleaner with nn.ModuleList or whatever
        #      put hidden_dim in params etc 
        zs = torch.split(z, self.split_zdim,
                         dim=-1)  # list of 4 [..., latent_dim/4]
        x = self.lin1(x)
        x = self.activ(x)
        x = torch.cat([x, zs[0]], dim=-1)
        x = self.lin2(x)
        x = self.activ(x)
        x = torch.cat([x, zs[1]], dim=-1)
        x = self.lin3(x)
        x = self.activ(x)
        x = torch.cat([x, zs[2]], dim=-1)
        x = self.lin4(x)
        x = self.activ(x)
        x = torch.cat([x, zs[3]], dim=-1)
        x = self.lin5(x)
        x = self.output_activ(x)
        return x


class HyperNetDecoder(Decoder):
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
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
    def __init__(self, latent_dim=64, input_dim=2, output_dim=3):
        super().__init__(latent_dim, input_dim, output_dim)

    def forward(self, x, z):
        return x * z


DECODERS = {
    "nerf": NeRFDecoder,
    # "linear": LinearDecoder,
    # "cat1st": Cat1stDecoder,
    # "distribcat": DistribCatDecoder,
    # "hypernet": HyperNetDecoder,
    # "attention": AttentionDecoder
}


class VANO(nn.Module):
    def __init__(self,
                 latent_dim=64,
                 input_dim=2,
                 output_dim=3,
                 decoder="nerf",
                 decoder_args={},
                 device='cpu'):
        super(VANO, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = Encoder(latent_dim, input_dim, output_dim)
        self.decoder = DECODERS[decoder](latent_dim, input_dim, output_dim,
                                         **decoder_args, device=device)

        ls_h = torch.linspace(0, 1, HEIGHT).to(device)
        ls_w = torch.linspace(0, 1, WIDTH).to(device)
        self.grid = torch.stack(
            torch.meshgrid(
                ls_h, ls_w,
                indexing='ij'),
            dim=-1).unsqueeze(0)

    def forward(self, u, custom_grid=None):
        """
        sample: use p(z) instead of p(z | u)
        custom_grid: use custom res (super or sub resolution)
        """

        mean, logvar = self.encoder(u)

        eps = torch.randn(u.shape[0], self.latent_dim, device=u.device)
        z = mean + eps * torch.exp(0.5 * logvar)

        grid = self.grid if custom_grid is None else custom_grid
        print(grid)
        print(grid.shape)
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


def load_data(N=1, case=1, device='cpu'):
    ls_h = torch.linspace(0, 1, HEIGHT).to(device)
    ls_w = torch.linspace(0, 1, WIDTH).to(device)
    grid = torch.stack(torch.meshgrid(ls_h, ls_w, indexing='ij'),
                       dim=-1)
    grid = torch.stack([grid] * N, dim=0)

    # Load MNIST
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = os.path.dirname(script_dir)
    data_dir = f'{script_dir}/data'
    u = tv.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
    )
    u = u.data.to(torch.float32) / 255.0
    u = u[:N]

    return grid.to(device), u.to(device)


def is_slurm():
    return shutil.which('sbatch') is not None


configs = [
    16,
]


@job(
    array=len(configs),
    partition="a5000,tesla,quadro,2080ti",
    cpus=4,
    gpus=1,
    ram="16GB",
    time="24:00:00",
    name="vanoceleba",
)
def train(i: int):
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():
    #    device = 'mps'
    else:
        device = 'cpu'

    trnsf = tv.transforms.Compose([
        #  transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # lambda x: x.float() / 255.0,
        #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # normalization
    ])

    batch_size = 2

    # Data
    N_train = 8192
    train_dataset = tv.datasets.CelebA(root='data', split='train',
                                       transform=trnsf)
    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  generator=torch.Generator(device=device))
    print(f"Train Data: {len(train_dataset)} samples, {len(train_loader)} "
          f"batches of size {batch_size}.")

    N_test = 256
    # TODO test and train data must be separate
    test_dataset = tv.datasets.CelebA(root='data', split='test',
                                      transform=trnsf)
    test_loader = tud.DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True,
                                 generator=torch.Generator(device=device))
    print(f"Test Data: {len(test_dataset)} samples, {len(test_loader)} "
          f"batches of size {batch_size}.")
    len_test_loader = min(len(test_loader), N_test // batch_size)

    torch.set_default_device(device)

    # Training
    decoder = 'nerf'
    vano = VANO(
        latent_dim=128,
        decoder=decoder,
        decoder_args=dict(
            use_pe=False,
            pe_m=0,
        ),
        device=device
    ).to(device)
    vano.train()

    # Parameters:
    S = 4  # Monte Carlo samples for evaluating reconstruction loss in ELBO (E_q(z | x) [log p(x | z)])
    # beta = 1e-5  # Weighting of KL divergence in ELBO
    beta = 1e-3  # β
    recon_reduction = 'mean'  # Reduction of reconstruction loss over grid points (mean or sum)
    num_iters = 25_000

    # Exponential decay of every 1000 iterations by 0.9
    lr = 1e-3
    lr_decay = 0.9
    lr_decay_every = 1000
    optimizer = topt.Adam(vano.parameters(), lr=lr)
    lr_scheduler = topt.lr_scheduler.StepLR(optimizer,
                                            step_size=lr_decay_every,
                                            gamma=lr_decay)

    # paper: Approximation of gaussian error in Banach spaces
    # mse: Typical mean squared error, as for finite data
    # ce: Cross-entropy loss, as for finite data (with Bernoulli assumption)
    recon_loss = 'mse'

    # W&B
    wandb_enabled = is_slurm()
    if wandb_enabled:
        wandb.init(
            project="vano",
            name=f"mse",
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
                "experiment-name": "celeba",
            }
        )

    len_train_loader = min(len(train_loader), N_train // batch_size)

    step = 0
    num_epochs = num_iters // len_train_loader
    print(f"Starting for {num_epochs} epochs")
    # num_epochs = max(num_epochs, 10)  # For experiment on N_train.
    for epoch in range(num_epochs):
        for _ in range(len_train_loader):
            u, _ = next(iter(train_loader))
            ls_h = torch.linspace(0, 1, HEIGHT).to(device)
            ls_w = torch.linspace(0, 1, WIDTH).to(device)
            grid = torch.stack(torch.meshgrid(ls_h, ls_w, indexing='ij'),
                               dim=-1).unsqueeze(0)
            grid = grid.expand(u.shape[0], *grid.shape[1:])

            u = u.to(device)
            grid = grid.to(device)
            # TODO: set_default_device

            mu, logvar, z, u_hat = vano(u.view(-1, 3, HEIGHT, WIDTH))
            u_hat = u_hat.squeeze()

            u, u_hat = u.flatten(1), u_hat.flatten(1)

            # Sample S values of z
            eps = torch.randn(S, *z.shape, device=z.device)
            z_samples = mu.unsqueeze(0) + eps * torch.exp(
                0.5 * logvar).unsqueeze(0)

            z_samples = z_samples.view(S * batch_size, *z_samples.shape[2:])
            u_hat_samples = vano.decoder(
                grid[:1].expand(z_samples.shape[0], *grid.shape[1:]),
                z_samples).squeeze()
            u_hat_samples = u_hat_samples.view(S, batch_size,
                                               *u_hat_samples.shape[1:])
            # u^: Shape=[4, bs, 64, 64, 3]
            u_hat_samples = u_hat_samples.flatten(start_dim=-3)
            # u^: Shape=[4, bs, 3*H*W]
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
                reconstr_loss = F.binary_cross_entropy(u_hat_samples,
                                                       u,
                                                       reduction='none')

            # Reduction
            if recon_reduction == 'mean':
                reconstr_loss = reconstr_loss.mean(
                    axis=-1)  # Mean over grid points
            elif recon_reduction == 'sum':
                reconstr_loss = reconstr_loss.sum(
                    axis=-1)  # Sum over grid points
            reconstr_loss = reconstr_loss.mean(axis=0)  # Mean over S
            reconstr_loss = reconstr_loss.mean(axis=0)  # Mean over batch

            kl_loss = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(
                axis=1).mean()

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

            if step % 1000 == 0:
                # if step % 100 == 0:
                with torch.no_grad():
                    vano.eval()

                    # ----- Reconstruction Image -----
                    test_u = test_dataset[0][0].to(device)
                    test_u_hat = vano(test_u.view(-1, 3, HEIGHT, WIDTH))[
                        3]
                    test_u_hat.squeeze_()
                    test_u.squeeze_()
                    test_u = test_u.permute(1, 2, 0)
                    reconstr_img = torch.cat([test_u,
                                              test_u_hat],
                                             axis=1).detach().cpu().numpy()
                    print((255 *
                           reconstr_img).astype(
                        np.uint8))
                    log_dict["reconstr_img"] = wandb.Image((255 *
                                                            reconstr_img).astype(
                        np.uint8))

                    # ----- Latent walk -----
                    u_start = test_dataset[0][0].to(device)
                    u_end = \
                        test_dataset[
                            torch.randint(0, len_test_loader, (1,),
                                          device='cpu')][
                            0].squeeze().to(device)
                    us = torch.stack([u_start, u_end]).view(-1, 3, HEIGHT,
                                                            WIDTH)
                    zs = vano.encoder(us)[0]
                    z_start = zs[0]
                    z_end = zs[1]
                    z_walk = torch.stack(
                        [z_start + (z_end - z_start) * (i / 10) for i in
                         range(10)], dim=0)
                    grids = vano.grid.expand(10, *vano.grid.shape[1:])
                    test_u_walk = vano.decoder(grids,
                                               z_walk).squeeze()  # [10, 218, 178, 3]
                    u_null = torch.zeros_like(u_start)
                    first_row = torch.cat([u_start] + 8 * [u_null] + [u_end],
                                          dim=2).permute(1, 2,
                                                         0)  # [218, 1780, 3]
                    # Display latent walk u's next to each other
                    second_row = test_u_walk.transpose(0, 1).reshape(HEIGHT,
                                                                     -1, 3)
                    latent_walk = torch.cat(
                        [first_row, second_row]).detach().cpu().numpy()
                    log_dict["latent_walk"] = wandb.Image((255 *
                                                           latent_walk).astype(
                        np.uint8))

                    # ----- Random sampling -----
                    img_grid_size = 10
                    # P(z) = N(0, I)
                    z = torch.randn(img_grid_size ** 2, vano.latent_dim).to(
                        device)
                    grids = vano.grid.expand(img_grid_size ** 2,
                                             *vano.grid.shape[1:])
                    us = vano.decoder(grids, z).squeeze()
                    us = us.view(img_grid_size, img_grid_size, HEIGHT,
                                 WIDTH, 3)
                    us = us.permute(0, 2, 1, 3, 4).reshape(
                        img_grid_size * HEIGHT, img_grid_size * WIDTH, 3)
                    rand_samples = us.detach().cpu().numpy()
                    log_dict["rand_samples"] = wandb.Image((255 *
                                                            rand_samples).astype(
                        np.uint8))

                    vano.train()

                    # ----- Super Res ----

                    resolutions = [
                        4, 8, 16, 28, 64, 128, 256, 512, 1024
                    ]
                    max_res = max(resolutions)
                    upsample = lambda x: F.interpolate(x,
                                                       size=(max_res,
                                                             max_res),
                                                       mode='nearest').squeeze()
                    empty = torch.zeros(3, max_res, max_res).detach().cpu()
                    test_u = test_dataset[0][0][None, ...].to(device)
                    multires_samples = [empty]
                    multires_decoded = [
                        upsample(test_u).squeeze().detach().cpu()]
                    z = torch.randn(1, vano.latent_dim).to(device)
                    for res in resolutions:
                        ls = torch.linspace(0, 1, res).to(device)
                        grid = torch.stack(
                            torch.meshgrid(ls, ls, indexing='ij'),
                            dim=-1).unsqueeze(0)
                        test_u_hat = vano(test_u, custom_grid=grid)[3]
                        sample_u_hat = vano.sample(device, z=z,
                                                   custom_grid=grid)

                        test_u_hat = test_u_hat.permute(0, 3, 1, 2)
                        sample_u_hat = sample_u_hat.permute(0, 3, 1, 2)

                        # Upsample with no interpolation to max_res
                        test_u_hat = upsample(test_u_hat)
                        sample_u_hat = upsample(sample_u_hat)

                        multires_samples.append(sample_u_hat.detach().cpu())
                        multires_decoded.append(test_u_hat.detach().cpu())

                    multires_samples = torch.cat(multires_samples, dim=2)
                    multires_decoded = torch.cat(multires_decoded, dim=2)

                    whole_grid = torch.cat(
                        [multires_samples, multires_decoded],
                        dim=1).permute(1, 2, 0).numpy()

                    log_dict["multires"] = wandb.Image(
                        (255 * whole_grid).astype(np.uint8))

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
