"""Preview images of the plate, its holes and the optimized profile."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import Grid
from .optimize import Result
from .profile import Profile


def _extent(grid: Grid):
    return [grid.x0, grid.x0 + grid.nx * grid.h, grid.y0, grid.y0 + grid.ny * grid.h]


def _size(grid: Grid, panels: int):
    w, h = grid.nx * grid.h, grid.ny * grid.h
    scale = 9.0 / max(w * panels, h)
    return (max(4.0, w * panels * scale + 1.5), max(4.0, h * scale + 1.0))


def _plate_edge(ax, grid: Grid):
    """Dashed outline of the original plate (padded so edges on the grid border close)."""
    m = np.pad((~grid.outside).astype(float), 1)
    xs = np.r_[grid.xc[0] - grid.h, grid.xc, grid.xc[-1] + grid.h]
    ys = np.r_[grid.yc[0] - grid.h, grid.yc, grid.yc[-1] + grid.h]
    ax.contour(xs, ys, m, levels=[0.5], colors=["tab:blue"], linewidths=0.6, linestyles="--")


def plot_holes(grid: Grid, path, fixed=(), loaded=()):
    fig, ax = plt.subplots(figsize=_size(grid, 1))
    img = np.where(grid.domain, 0.25, np.nan)
    ax.imshow(img, origin="lower", extent=_extent(grid), cmap="Greys", vmin=0, vmax=1)
    for hl in grid.holes:
        color = "tab:blue" if hl.name in fixed else "tab:red" if hl.name in loaded else "k"
        ax.contour(grid.xc, grid.yc, hl.mask.astype(float), levels=[0.5], colors=[color], linewidths=1)
        ax.annotate(hl.name, hl.centroid, xytext=(5, 5), textcoords="offset points", fontsize=9, color=color,
                    fontweight="bold", bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_aspect("equal")
    ax.set_title("Holes (blue = fixed, red = loaded)")
    ax.set_xlabel("mm")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_result(grid: Grid, result: Result, profile: Profile, path, fixed=(), loaded=()):
    fig, axs = plt.subplots(1, 2, figsize=_size(grid, 2))
    axs[0].imshow(result.density, origin="lower", extent=_extent(grid), cmap="Greys", vmin=0, vmax=1)
    axs[0].set_title("Optimized material layout")
    ax = axs[1]
    ax.fill(*profile.outline.T, color="#e8a33d", ec="k", lw=0.6)
    for q in profile.pockets:
        ax.fill(*q.T, color="white", ec="k", lw=0.6)
    for a in axs:
        _plate_edge(a, grid)
        for hl in grid.holes:
            color = "tab:blue" if hl.name in fixed else "tab:red" if hl.name in loaded else "k"
            a.contourf(grid.xc, grid.yc, hl.mask.astype(float), levels=[0.5, 1.5], colors=["white"])
            a.contour(grid.xc, grid.yc, hl.mask.astype(float), levels=[0.5], colors=[color], linewidths=0.9)
        a.set_aspect("equal")
        a.set_xlim(grid.x0 - 3, grid.x0 + grid.nx * grid.h + 3)
        a.set_ylim(grid.y0 - 3, grid.y0 + grid.ny * grid.h + 3)
    ax.set_title("Cut profile (dashed = original plate)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
