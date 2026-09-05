"""Shared figure style, close to the one used in the paper."""

import matplotlib as mpl
import matplotlib.pyplot as plt

STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
}


def use_paper_style():
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(STYLE)


def show_field(ax, field, cmap="viridis", **kwargs):
    """Draw a ``(nx, ny)`` field with x along the horizontal axis."""
    image = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                      interpolation="nearest", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def grid(nrows, ncols, size=2.0, **kwargs):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(size * ncols, size * nrows), **kwargs)
    return fig, axes.reshape(nrows, ncols)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
