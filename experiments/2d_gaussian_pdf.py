import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as topt
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# torch distribs
import torch.distributions as dists

from dawgz import job, after, ensure, schedule

import wandb

class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1, device='cpu'):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activ = nn.GELU()

        # Input: [output_dim, 48, 48]
        self.seq = nn.Sequential(
            nn.Conv2d(output_dim, 8, 2),        # [8, 47, 47]
            self.activ,
            nn.Conv2d(8, 16, 2),                # [16, 46, 46]
            self.activ,
            nn.MaxPool2d(2),                    # [16, 23, 23]
            nn.Conv2d(16, 32, 2),               # [32, 22, 22]
            self.activ,
            nn.Conv2d(32, 64, 2),               # [64, 21, 21]
            self.activ,
            nn.MaxPool2d(2),                    # [64, 10, 10]
            nn.Flatten(),                       # [64 * 10 * 10]
            nn.Linear(64 * 10 * 10, 256),       # [256]
            self.activ,
            nn.Linear(256, 128),                # [128]
            self.activ,
            nn.Linear(128, 2 * self.latent_dim) # [2 * latent_dim]
        )

    def forward(self, u):
        out = self.seq(u)
        mean = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1, device='cpu'):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # (original) NeRF-like architecture
        self.mlp_x = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(),
            nn.Linear(2 * self.latent_dim, 32)
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x, z):
        """
        Computes u(x) by conditioning on z, a latent representation of u.

        Args:
            x: (batch_size, input_dim) tensor of spatial locations
            z: (batch_size, latent_dim) tensor of latent representations
        """
        x = self.mlp_x(x)
        z = self.mlp_z(z)
        # z is [32, 32], reshape to [32, 48, 48, 32]
        z = z.view(-1, 1, 1, 32).expand(-1, 48, 48, -1)
        xz = torch.cat([x, z], dim=-1)
        xz = self.joint_mlp(xz)

        return xz


class VANO(nn.Module):
    def __init__(self, latent_dim=32, input_dim=2, output_dim=1, device='cpu'):
        super(VANO, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
    
        self.encoder = Encoder(latent_dim, input_dim, output_dim, device)
        self.decoder = Decoder(latent_dim, input_dim, output_dim, device)

        ls = torch.linspace(0, 1, 48).to(device)
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
    

def gen_datasets(N=1, device='cpu'):
    """
    Creates N 2D gaussian pdfs.
    """
    mu_x = dists.Uniform(0, 1).sample((N,))
    mu_y = dists.Uniform(0, 1).sample((N,))

    # Std dev
    sigma = 0.01 + dists.Uniform(0, 0.1).sample((N,))
    
    dim_range = torch.linspace(0, 1, 48)
    grid = torch.stack(torch.meshgrid(dim_range, dim_range, indexing='ij'), dim=-1)

    x = torch.stack([grid] * N, dim=0)
    y = torch.stack([dists.MultivariateNormal(
        torch.tensor([mu_x[i], mu_y[i]]),
        covariance_matrix=(sigma[i]**2 * torch.eye(2))
        ).log_prob(grid) for i in range(N)], dim=0)

    return x.to(device), y.exp().to(device)

@job(
    partition="a5000,tesla,quadro",
    cpus=4,
    gpus=1,
    ram="16GB",
    time="24:00:00",
    name="vano",
)
def train():
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Data
    N_train = 2048
    train_data = gen_datasets(N_train, device=device)
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    N_test = 128
    test_data = gen_datasets(N_test, device=device)
    test_dataset = torch.utils.data.TensorDataset(*test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Training
    vano = VANO(device=device).to(device)
    vano.train()

    # Parameters:
    S = 4  # Monte Carlo samples for evaluating reconstruction loss in ELBO (E_q(z | x) [log p(x | z)])
    beta = 10e-5  # Weighting of KL divergence in ELBO
    batch_size = 32
    num_iters = 20_000

    # Exponential decay of every 1000 iterations by 0.9
    lr = 1e-3
    lr_decay = 0.9
    lr_decay_every = 1000
    optimizer = topt.Adam(vano.parameters(), lr=lr)
    lr_scheduler = topt.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)

    # W&B
    wandb_enabled = True
    if wandb_enabled:
        wandb.init(
            project="vano",
            name="vano",
            config={
                "S": S,
                "beta": beta,
                "batch_size": batch_size,
                "num_iters": num_iters,
                "lr": lr,
                "lr_decay": lr_decay,
                "lr_decay_every": lr_decay_every
            }
        )

    step = 0
    num_epochs = num_iters // len(train_loader)
    for epoch in range(num_epochs):
        for grid, u in train_loader:
            mu, logvar, z, u_hat = vano(u.view(-1, 1, 48, 48))
            u_hat = u_hat.squeeze()

            u, u_hat = u.flatten(1), u_hat.flatten(1)

            # ELBO = E_p(eps)[log p(x | z=g(eps, x))] - KL(q(z | x) || p(z))
            reconstr_loss = F.mse_loss(u_hat, u, reduction='none').sum(axis=1).mean()
            #kl_loss = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(axis=1).mean()

            loss = reconstr_loss #+ beta * kl_loss
            
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            lr_scheduler.step()

            log_dict = {
                "reconstr_loss": reconstr_loss.item(),
                #"kl_loss": kl_loss.item(),
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }

            if step % 1000 == 0:
                with torch.no_grad():
                    # Reconstruction Image
                    test_u = test_dataset[0][1]
                    test_u_hat = vano(test_u.view(-1, 1, 48, 48))[3].squeeze()
                    reconstr_img = torch.cat([test_u, test_u_hat], axis=1).detach().cpu().numpy()
                    log_dict["reconstr_img"] = wandb.Image(reconstr_img)

                    # Latent walk
                    test_uS = test_dataset[0][1]
                    test_uE = test_dataset[torch.randint(0, len(test_dataset), (1,))][1]
                    z_start = vano.encoder(test_u.view(-1, 1, 48, 48))[0]  # Mean of q(z1 | x)
                    z_end = vano.encoder(test_u.view(-1, 1, 48, 48))[0]  # Mean of q(z2 | x)
                    print(z_start.shape, z_end.shape)
                    z_walk = torch.stack([z_start + (z_end - z_start) * (i / 10) for i in range(10)], dim=0)
                    print(z_walk.shape)
                    test_u_walk = vano.decoder(vano.grid.expand(10, *vano.grid.shape[1:]), z_walk).squeeze()
                    print(vano.grid.expand(10, *vano.grid.shape[1:]).shape)
                    print(vano.decoder(vano.grid.expand(10, *vano.grid.shape[1:]), z_walk).shape)
                    latent_walk = torch.cat([test_uS, test_u_walk, test_uE], axis=1).detach().cpu().numpy()
                    print(latent_walk.shape)
                    log_dict["latent_walk"] = wandb.Image(latent_walk)
                    print()



            if wandb_enabled:
                wandb.log(log_dict, step=step)

            step += 1

    # Save model
    torch.save(vano.state_dict(), "vano.pt")

if __name__ == "__main__":
    schedule(
        train,
        backend="slurm",
        export="ALL",
        env=["export WANDB_SILENT=true"],
    )