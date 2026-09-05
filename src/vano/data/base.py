"""Container for a set of functions observed on a shared grid."""

from dataclasses import dataclass

import torch


def unit_grid(resolution, dim=2, device=None, dtype=torch.float32):
    """Query coordinates on ``[0, 1]^dim``, ordered so that axis 0 is x.

    Returns ``(P, dim)`` with ``P = resolution ** dim``.  The flattening is
    row-major in that order, so ``coords.reshape(res, ..., dim)[i, j]`` is
    ``(x_i, y_j)`` and a field can be reshaped to an image the same way.
    """
    axis = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    mesh = torch.meshgrid(*([axis] * dim), indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, dim)


@dataclass
class FunctionData:
    """``N`` functions: encoder inputs ``u``, targets ``s`` on the grid ``y``.

    ``w`` is the per-function weight of the reconstruction term.  The release
    calls it "empirical norm rescaling": on the benchmarks whose functions vary
    in magnitude by orders of magnitude (GRF, 2D Gaussian densities) it is
    ``1 / ||u||_2^2``, so that every function contributes equally; elsewhere it
    is 1.
    """

    u: torch.Tensor          # (N, *input_shape)
    y: torch.Tensor          # (P, query_dim), shared by all functions
    s: torch.Tensor          # (N, P, out_dim)
    w: torch.Tensor          # (N,)
    grid_shape: tuple        # spatial shape the flat P axis unfolds to

    def __len__(self):
        return self.u.shape[0]

    def to(self, device):
        return FunctionData(self.u.to(device), self.y.to(device),
                            self.s.to(device), self.w.to(device),
                            self.grid_shape)

    def __getitem__(self, idx):
        return FunctionData(self.u[idx], self.y, self.s[idx], self.w[idx],
                            self.grid_shape)
