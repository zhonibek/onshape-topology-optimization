"""2D SIMP topology optimization of a flat plate (plane stress, extrusion constrained).

Minimises the weighted compliance of several in-plane load cases under a
material budget. Screw holes keep a solid ring; one set of holes is held
fixed and loads are applied at another set of holes.

Method: bilinear quad elements, density filter + Heaviside projection with
beta continuation, optimality-criteria updates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Grid

LOAD_CASES = ("x", "y", "twist")


@dataclass
class Settings:
    volfrac: float = 0.40        # share of the plate's material to keep
    rmin: float = 4.5            # filter radius in mm (bigger -> thicker members)
    ring: float = 3.0            # solid wall kept around every screw hole, mm
    penal: float = 3.0
    nu: float = 0.3
    emin: float = 1e-9
    max_iter: int = 260
    beta_max: float = 32.0
    beta_every: int = 50
    load_cases: tuple[str, ...] = LOAD_CASES


@dataclass
class Result:
    density: np.ndarray                  # (ny, nx), NaN outside material
    ring: np.ndarray                     # (ny, nx) bool
    iterations: int
    volume_fraction: float
    stiffness_retained: dict[str, float]
    history: list = field(default_factory=list)
    runtime_s: float = 0.0


def element_stiffness(nu: float) -> np.ndarray:
    """8x8 stiffness of a square Q4 element (E = 1, unit thickness; size-independent)."""
    D = 1 / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    KE = np.zeros((8, 8))
    g = 1 / np.sqrt(3)
    nat = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for xi in (-g, g):
        for eta in (-g, g):
            dN = np.array([[n[0] * (1 + eta * n[1]) / 4, n[1] * (1 + xi * n[0]) / 4] for n in nat]) * 2
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dN[:, 0], dN[:, 1]
            B[2, 0::2], B[2, 1::2] = dN[:, 1], dN[:, 0]
            KE += B.T @ D @ B / 4
    return KE


def ring_mask(grid: Grid, ring: float, holes=None) -> np.ndarray:
    any_hole = np.zeros_like(grid.domain)
    for hl in grid.holes if holes is None else holes:
        any_hole |= hl.mask
    if not any_hole.any():
        return np.zeros_like(grid.domain)
    dist = ndi.distance_transform_edt(~any_hole) * grid.h
    return grid.domain & (dist <= ring + grid.h / 2)


def optimize(grid: Grid, fixed: list[str], loaded: list[str], settings: Settings | None = None, log=print) -> Result:
    s = settings or Settings()
    if not fixed or not loaded:
        raise ValueError("Pick at least one fixed hole and one loaded hole.")
    if set(fixed) & set(loaded):
        raise ValueError("A hole cannot be both fixed and loaded.")
    t0 = time.time()
    nx, ny, h = grid.nx, grid.ny, grid.h
    active = grid.domain.ravel()
    ring = ring_mask(grid, s.ring).ravel()
    ae = np.flatnonzero(active)
    na = ae.size
    ring_a = ring[ae]
    design_a = ~ring_a
    if design_a.sum() == 0:
        raise ValueError("No design area left after the screw-hole rings.")
    if ring_a.sum() / na >= s.volfrac:
        raise ValueError(
            f"The screw-hole rings alone use {ring_a.sum() / na:.0%} of the plate; raise --volfrac or lower --ring."
        )

    KE = element_stiffness(s.nu)
    ii, jj = ae % nx, ae // nx
    n0 = jj * (nx + 1) + ii
    nodes = np.stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1], axis=1)
    edof = np.empty((na, 8), dtype=np.int64)
    edof[:, 0::2], edof[:, 1::2] = 2 * nodes, 2 * nodes + 1
    ndof = 2 * (nx + 1) * (ny + 1)
    node_x = grid.x0 + (np.arange((nx + 1) * (ny + 1)) % (nx + 1)) * h
    node_y = grid.y0 + (np.arange((nx + 1) * (ny + 1)) // (nx + 1)) * h
    used_nodes = np.unique(nodes)

    def boundary_nodes(name):
        he = np.flatnonzero(grid.hole(name).mask.ravel())
        hn0 = (he // nx) * (nx + 1) + he % nx
        hn = np.unique(np.concatenate([hn0, hn0 + 1, hn0 + nx + 2, hn0 + nx + 1]))
        return np.intersect1d(hn, used_nodes)

    fixed_nodes = np.concatenate([boundary_nodes(k) for k in fixed])
    fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    free = np.setdiff1d(np.unique(edof), fixed_dofs)
    red = -np.ones(ndof, dtype=np.int64)
    red[free] = np.arange(free.size)
    nf = free.size

    # load cases on the loaded holes (each hole gets an equal share)
    load_nodes = {k: boundary_nodes(k) for k in loaded}
    cases = [c for c in s.load_cases if c in LOAD_CASES]
    F = np.zeros((ndof, len(cases)))
    alln = np.concatenate(list(load_nodes.values()))
    cx, cy = node_x[alln].mean(), node_y[alln].mean()
    for c, case in enumerate(cases):
        if case in ("x", "y"):
            off = 0 if case == "x" else 1
            for ns in load_nodes.values():
                F[2 * ns + off, c] += 1.0 / len(loaded) / ns.size
        else:  # unit twisting moment about the loaded group's centre
            rx, ry = node_x[alln] - cx, node_y[alln] - cy
            alpha = 1.0 / max(np.sum(rx**2 + ry**2), 1e-12)
            np.add.at(F[:, c], 2 * alln, -alpha * ry)
            np.add.at(F[:, c], 2 * alln + 1, alpha * rx)
    Fr = F[free]

    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    riK, rjK = red[iK], red[jK]
    keep = (riK >= 0) & (rjK >= 0)
    riK, rjK = riK[keep], rjK[keep]
    KEf = KE.ravel()

    def solve(E):
        sK = (KEf[None, :] * E[:, None]).ravel()[keep]
        K = sp.coo_matrix((sK, (riK, rjK)), shape=(nf, nf)).tocsc()
        U = np.zeros((ndof, len(cases)))
        U[free] = spla.splu(K, permc_spec="MMD_AT_PLUS_A").solve(Fr)
        return U

    def compliance(U):
        return np.einsum("ij,ij->j", F, U)

    # density filter restricted to the plate
    grid_to_a = -np.ones(nx * ny, dtype=np.int64)
    grid_to_a[ae] = np.arange(na)
    r = s.rmin / h
    R = int(np.ceil(r)) - 1
    rows, cols, vals = [], [], []
    for dj in range(-R, R + 1):
        for di in range(-R, R + 1):
            w = r - np.hypot(di, dj)
            if w <= 0:
                continue
            i2, j2 = ii + di, jj + dj
            ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny)
            tgt = np.full(na, -1)
            tgt[ok] = grid_to_a[j2[ok] * nx + i2[ok]]
            ok = tgt >= 0
            rows.append(np.flatnonzero(ok))
            cols.append(tgt[ok])
            vals.append(np.full(ok.sum(), w))
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(na, na)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    eta = 0.5

    def physical(x, beta):
        xt = (H @ x) / Hs
        den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        xp = (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den
        dxp = beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / den
        xp[ring_a], dxp[ring_a] = 1.0, 0.0
        return xp, dxp

    log(f"grid {nx}x{ny} at {h} mm: {na} elements, {ring_a.sum()} in screw rings, {nf} unknowns")
    x = np.ones(na)
    x[design_a] = (s.volfrac * na - ring_a.sum()) / design_a.sum()
    beta = 1.0
    U = solve(s.emin + physical(x, beta)[0] ** s.penal * (1 - s.emin))
    w = 1.0 / compliance(U)            # every load case counts equally

    hist, loop, since = [], 0, 0
    while loop < s.max_iter:
        loop += 1
        since += 1
        xp, dxp = physical(x, beta)
        U = solve(s.emin + xp**s.penal * (1 - s.emin))
        ce = np.stack([np.einsum("ij,jk,ik->i", U[edof, c], KE, U[edof, c]) for c in range(len(cases))], axis=1)
        obj = float(np.sum(w * compliance(U)))
        dc = H.T @ (-s.penal * xp ** (s.penal - 1) * (1 - s.emin) * (ce @ w) * dxp / Hs)
        dv = H.T @ (dxp / na / Hs)
        dcd = np.minimum(dc[design_a], -1e-30)
        dvd = np.maximum(dv[design_a], 1e-30)
        xd = x[design_a]
        move = 0.2 if beta <= 2 else 0.1
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(xd * np.sqrt(-dcd / dvd / lm), np.maximum(0, xd - move), np.minimum(1, xd + move))
            xt = x.copy()
            xt[design_a] = xn
            if physical(xt, beta)[0].mean() > s.volfrac:
                l1 = lm
            else:
                l2 = lm
        change = float(np.max(np.abs(xn - xd)))
        x[design_a] = xn
        hist.append((loop, beta, obj, float(xp.mean()), change))
        if loop == 1 or loop % 20 == 0:
            log(f"  it {loop:3d}  beta {beta:4.0f}  objective {obj:.4f}  material {xp.mean():.3f}  ({time.time() - t0:.0f}s)")
        if beta < s.beta_max and (since >= s.beta_every or change < 0.01):
            beta, since = beta * 2, 0
        elif beta >= s.beta_max and change < 0.01:
            break

    xp, _ = physical(x, beta)
    binary = np.where(ring_a, 1.0, (xp >= 0.5).astype(float))
    c_full = compliance(solve(np.ones(na)))
    c_bin = compliance(solve(s.emin + binary * (1 - s.emin)))
    density = np.full(nx * ny, np.nan)
    density[ae] = xp
    return Result(
        density=density.reshape(ny, nx),
        ring=ring.reshape(ny, nx),
        iterations=loop,
        volume_fraction=float(binary.mean()),
        stiffness_retained={c: float(c_full[i] / c_bin[i]) for i, c in enumerate(cases)},
        history=hist,
        runtime_s=time.time() - t0,
    )
