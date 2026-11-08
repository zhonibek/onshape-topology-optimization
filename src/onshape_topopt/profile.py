"""Turn an optimized density field into CAD loops: one outline and a set of pockets."""
from __future__ import annotations

from dataclasses import dataclass, field

import contourpy
import numpy as np
import scipy.ndimage as ndi
from matplotlib.path import Path

from .geometry import Grid, Plate, rasterize_triangles
from .optimize import Result


@dataclass
class ProfileSettings:
    extend: float = 2.0          # mm the outline is pushed past kept edges (so they stay exact)
    field_smoothing: float = 0.7  # Gaussian sigma in elements before contouring
    pocket_smoothing: float = 0.8  # Gaussian sigma in mm along each pocket loop (0 = off)
    min_pocket_area: float = 8.0  # mm^2; smaller pockets are dropped
    simplify: float = 0.03       # mm, Douglas-Peucker tolerance


@dataclass
class Profile:
    outline: np.ndarray                       # (k, 2) mm, plate frame
    pockets: list[np.ndarray]
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def polygon_area(p: np.ndarray) -> float:
    q = np.vstack([p, p[:1]])
    return 0.5 * float(np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1]))


def ccw(p: np.ndarray) -> np.ndarray:
    return p if polygon_area(p) > 0 else p[::-1]


def _dp(points: np.ndarray, tol: float) -> np.ndarray:
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        L = np.hypot(*seg)
        rel = points[a + 1:b] - points[a]
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L if L > 0 else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return points[keep]


def simplify_loop(p: np.ndarray, tol: float) -> np.ndarray:
    p = p[:-1] if np.allclose(p[0], p[-1]) else p
    k = int(np.argmax(np.hypot(*(p - p[0]).T)))
    a = _dp(p[:k + 1], tol)
    b = _dp(np.vstack([p[k:], p[:1]]), tol)
    return np.vstack([a, b[1:-1]])


def resample(p: np.ndarray, step: float) -> np.ndarray:
    q = np.vstack([p, p[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    t = np.arange(0, s[-1], step)
    return np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]


def smooth_loop(p: np.ndarray, sigma_mm: float, step: float = 0.2) -> np.ndarray:
    r = resample(p, step)
    sg = sigma_mm / step
    return np.c_[ndi.gaussian_filter1d(r[:, 0], sg, mode="wrap"), ndi.gaussian_filter1d(r[:, 1], sg, mode="wrap")]


def _densify(p: np.ndarray, step: float = 0.2) -> np.ndarray:
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * np.arange(n)[:, None] / n)
    return np.vstack(out)


def _point_to_loop(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    B = np.roll(A, -1, axis=0)
    AB = B - A
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / np.maximum((AB**2).sum(-1)[None], 1e-12), 0, 1)
    C = A[None] + t[..., None] * AB[None]
    return np.min(np.hypot(*(P[:, None, :] - C).transpose(2, 0, 1)), axis=1)


def extract_profile(grid: Grid, result: Result, plate: Plate, ps: ProfileSettings | None = None) -> Profile:
    ps = ps or ProfileSettings()
    h = grid.h
    field_ = np.nan_to_num(result.density, nan=0.0)
    holes_any = np.zeros_like(grid.domain)
    for hl in grid.holes:
        holes_any |= hl.mask
    field_[holes_any] = 1.0                       # holes already exist in the CAD
    inside = grid.domain | holes_any

    P = int(np.ceil((ps.extend + 3.0) / h))
    fe = np.pad(field_, P)
    dom = np.pad(inside, P)
    dist, (ni, nj) = ndi.distance_transform_edt(~dom, return_indices=True)
    fe = np.where(dom, fe, fe[ni, nj])            # continue the edge value outward ...
    fe[(~dom) & (dist * h > ps.extend)] = 0.0     # ... for `extend` mm, then empty
    fe = ndi.gaussian_filter(fe, sigma=ps.field_smoothing)
    xg = grid.x0 + (np.arange(-P, grid.nx + P) + 0.5) * h
    yg = grid.y0 + (np.arange(-P, grid.ny + P) + 0.5) * h

    gen = contourpy.contour_generator(xg, yg, fe, fill_type=contourpy.FillType.OuterOffset)
    pts_list, off_list = gen.filled(0.5, 10.0)
    pieces = []
    for pts, off in zip(pts_list, off_list):
        loops = [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        pieces.append((abs(polygon_area(loops[0])), loops[0], loops[1:]))
    pieces.sort(key=lambda t: -t[0])
    warnings = []
    if len(pieces) > 1 and pieces[1][0] > 4.0:
        warnings.append(f"design has {len(pieces)} separate solid pieces; only the largest is kept")
    _, outer, inner = pieces[0]

    pockets = [q for q in inner if abs(polygon_area(q)) >= ps.min_pocket_area]
    if ps.pocket_smoothing > 0:
        pockets = [smooth_loop(q, ps.pocket_smoothing) for q in pockets]
    outline = ccw(simplify_loop(outer, ps.simplify))
    pockets = [ccw(simplify_loop(q, ps.simplify)) for q in pockets]

    # every hole must sit inside the solid and outside every pocket
    X, Y = np.meshgrid(grid.xc, grid.yc)
    op = Path(outline)
    min_wall = np.inf
    for hl in grid.holes:
        c = hl.centroid
        if not op.contains_point(c) or any(Path(q).contains_point(c) for q in pockets):
            warnings.append(f"hole {hl.name} is not surrounded by material")
        pts = np.c_[X[hl.mask], Y[hl.mask]]
        for q in pockets:
            d = _point_to_loop(pts, q).min() - h / 2
            min_wall = min(min_wall, d)

    # final solid on a fine grid: inside the original plate, inside the outline, outside pockets
    step = h / 4
    xf = grid.x0 + (np.arange(grid.nx * 4) + 0.5) * step
    yf = grid.y0 + (np.arange(grid.ny * 4) + 0.5) * step
    fine = rasterize_triangles(plate.triangles, xf, yf)
    XF, YF = np.meshgrid(xf, yf)
    pts_all = np.c_[XF.ravel(), YF.ravel()]
    in_outline = op.contains_points(pts_all).reshape(fine.shape)
    in_pocket = np.zeros_like(fine)
    for q in pockets:
        in_pocket |= Path(q).contains_points(pts_all).reshape(fine.shape)
    solid = fine & in_outline & ~in_pocket
    outer_empty = ~solid & ~in_pocket
    dist_outer = ndi.distance_transform_edt(~outer_empty) * step

    webs = []
    dens = [_densify(q) for q in pockets]
    for i, Pi in enumerate(dens):
        # member between this pocket and the plate edge / screw holes
        ix = np.clip(np.round((Pi[:, 0] - xf[0]) / step).astype(int), 0, len(xf) - 1)
        iy = np.clip(np.round((Pi[:, 1] - yf[0]) / step).astype(int), 0, len(yf) - 1)
        webs.append(float(dist_outer[iy, ix].min()))
        for j in range(i + 1, len(pockets)):
            webs.append(float(_point_to_loop(Pi, pockets[j]).min()))
    openings = []
    for q in pockets:
        Pq = _densify(q)
        n = len(Pq)
        D = np.hypot(Pq[:, None, 0] - Pq[None, :, 0], Pq[:, None, 1] - Pq[None, :, 1])
        idx = np.arange(n)
        gap = np.minimum(np.abs(idx[:, None] - idx[None]), n - np.abs(idx[:, None] - idx[None]))
        D[gap < 40] = np.inf
        openings.append(float(D.min()) if np.isfinite(D).any() else float("nan"))

    # predicted areas, used to check the CAD after each Onshape step
    plate_area = plate.area_mm2()
    trimmed_area = float((fine & in_outline).sum() * step * step)
    pocket_area = float(sum(polygon_area(q) for q in pockets))
    final_area = float(solid.sum() * step * step)

    stats = {
        "plate_area_mm2": plate_area,
        "trimmed_area_mm2": trimmed_area,
        "pocket_area_mm2": pocket_area,
        "final_area_mm2": final_area,
        "final_material_fraction": final_area / plate_area,
        "pockets": len(pockets),
        "outline_points": len(outline),
        "pocket_points": int(sum(len(q) for q in pockets)),
        "thinnest_member_mm": min(webs) if webs else None,
        "narrowest_pocket_mm": min(openings) if openings else None,
        "min_hole_wall_mm": None if not np.isfinite(min_wall) else float(min_wall),
    }
    return Profile(outline, pockets, stats, warnings)
