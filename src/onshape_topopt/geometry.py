"""Turn Onshape face triangles into a flat plate, a material grid and its holes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

PARALLEL = 0.9999   # |cos| above this counts as parallel to the plate normal
PERPENDICULAR = 1e-3


class NotAPlateError(ValueError):
    """The selected parts are not one flat plate of constant thickness."""


@dataclass
class Plate:
    """A flat plate. 2D coordinates are in mm in the (u, v) frame on the top face."""

    normal: np.ndarray          # unit vector, bottom -> top
    u: np.ndarray               # in-plane axes (world)
    v: np.ndarray
    top: float                  # top face height along normal (m)
    bottom: float
    triangles: np.ndarray       # (k, 3, 2) top-face triangles, mm
    part_ids: list[str] = field(default_factory=list)

    @property
    def thickness_mm(self) -> float:
        return (self.top - self.bottom) * 1000.0

    @property
    def origin(self) -> np.ndarray:
        return self.normal * self.top

    def to_world(self, pts_mm, height: float | None = None) -> np.ndarray:
        """Plate (u, v) mm -> world metres, on the top face (or at `height` along the normal)."""
        p = np.asarray(pts_mm, float) / 1000.0
        h = self.top if height is None else height
        return self.normal * h + p[..., :1] * self.u + p[..., 1:2] * self.v

    def area_mm2(self) -> float:
        t = self.triangles
        a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        return float(np.sum(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])) / 2)


def _facets(tess, part_ids=None):
    for body in tess:
        if part_ids and body.get("id") not in part_ids:
            continue
        if body.get("bodyType", "solid") != "solid":
            continue
        for face in body.get("faces", []):
            for f in face.get("facets", []):
                yield np.asarray(f["normal"], float), np.asarray(f["vertices"], float)


def plate_from_tessellation(tess, part_ids=None, tol: float = 2e-5) -> Plate:
    """Build a Plate from GET .../tessellatedfaces output (coordinates in metres)."""
    facets = list(_facets(tess, part_ids))
    if not facets:
        raise NotAPlateError("No solid faces found for the selected parts.")
    normals = np.array([n / np.linalg.norm(n) for n, _ in facets])
    areas = np.array([np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])) / 2 for _, v in facets])

    # plate normal = direction carrying the most face area (top + bottom)
    best, best_area = None, -1.0
    for n in normals:
        a = areas[np.abs(normals @ n) > PARALLEL].sum()
        if a > best_area:
            best, best_area = n, a
    d = best.copy()
    cos = normals @ d
    bad = (np.abs(cos) < PARALLEL) & (np.abs(cos) > PERPENDICULAR)
    if bad.any():
        raise NotAPlateError(
            f"{bad.sum()} face triangles are neither parallel nor perpendicular to the plate. "
            "Only flat, extruded plates are supported (no chamfers, fillets on faces, or slopes)."
        )
    top_mask, bot_mask = cos > PARALLEL, cos < -PARALLEL
    top_h = np.array([v @ d for (_, v), m in zip(facets, top_mask) if m]).ravel()
    bot_h = np.array([v @ d for (_, v), m in zip(facets, bot_mask) if m]).ravel()
    if top_h.size == 0 or bot_h.size == 0:
        raise NotAPlateError("Could not find both a top and a bottom face.")
    if np.ptp(top_h) > tol or np.ptp(bot_h) > tol:
        raise NotAPlateError(
            "Top or bottom faces are at different heights: the parts must form one plate "
            "of constant thickness (pockets, steps or bosses are not supported yet)."
        )
    top, bottom = float(top_h.mean()), float(bot_h.mean())

    # in-plane frame: world axis least aligned with the normal
    axes = np.eye(3)
    a = axes[int(np.argmin(np.abs(axes @ d) + np.array([0, 1e-9, 2e-9])))]
    u = a - (a @ d) * d
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    origin = d * top
    tris = np.array([v_ for (_, v_), m in zip(facets, top_mask) if m])
    rel = tris - origin
    tri2d = np.stack([rel @ u, rel @ v], axis=-1) * 1000.0
    return Plate(d, u, v, top, bottom, tri2d, list(part_ids or []))


# ---------------------------------------------------------------- grid
@dataclass
class Hole:
    name: str
    mask: np.ndarray                # (ny, nx) bool, elements inside the hole
    centroid: tuple[float, float]   # plate mm
    size: tuple[float, float]       # bounding box width, height (mm)
    area_mm2: float = 0.0


@dataclass
class Grid:
    h: float
    x0: float
    y0: float
    nx: int
    ny: int
    domain: np.ndarray              # (ny, nx) bool: material elements
    holes: list[Hole]

    @property
    def xc(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.h

    @property
    def yc(self) -> np.ndarray:
        return self.y0 + (np.arange(self.ny) + 0.5) * self.h

    def hole(self, name: str) -> Hole:
        for hl in self.holes:
            if hl.name.lower() == name.lower():
                return hl
        raise KeyError(f"No hole called {name}; holes are {', '.join(h.name for h in self.holes)}")

    @property
    def outside(self) -> np.ndarray:
        """Elements outside the plate outline (not material and not a hole)."""
        any_hole = np.zeros_like(self.domain)
        for hl in self.holes:
            any_hole |= hl.mask
        return ~self.domain & ~any_hole


def rasterize_triangles(tri2d: np.ndarray, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Boolean mask of grid points (centres) covered by any triangle."""
    X, Y = np.meshgrid(xc, yc)
    mask = np.zeros(X.shape, bool)
    eps = 1e-9
    for a, b, c in tri2d:
        lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        i0, i1 = np.searchsorted(xc, lo[0]), np.searchsorted(xc, hi[0], side="right")
        j0, j1 = np.searchsorted(yc, lo[1]), np.searchsorted(yc, hi[1], side="right")
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = X[j0:j1, i0:i1], Y[j0:j1, i0:i1]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
        l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
        l3 = 1 - l1 - l2
        mask[j0:j1, i0:i1] |= (l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps)
    return mask


def make_grid(plate: Plate, h: float) -> Grid:
    t = plate.triangles.reshape(-1, 2)
    lo, hi = t.min(axis=0), t.max(axis=0)
    nx, ny = int(np.ceil((hi[0] - lo[0]) / h)), int(np.ceil((hi[1] - lo[1]) / h))
    x0, y0 = lo[0] - (nx * h - (hi[0] - lo[0])) / 2, lo[1] - (ny * h - (hi[1] - lo[1])) / 2
    xc, yc = x0 + (np.arange(nx) + 0.5) * h, y0 + (np.arange(ny) + 0.5) * h
    domain = rasterize_triangles(plate.triangles, xc, yc)

    # holes = empty regions fully enclosed by material
    padded, n = ndi.label(np.pad(~domain, 1, constant_values=True))
    border = set(np.unique(np.concatenate([padded[0], padded[-1], padded[:, 0], padded[:, -1]]))) | {0}
    labels = padded[1:-1, 1:-1]
    found = []
    X, Y = np.meshgrid(xc, yc)
    for k in range(1, n + 1):
        if k in border:
            continue
        m = labels == k
        cx, cy = float(X[m].mean()), float(Y[m].mean())
        w = float(X[m].max() - X[m].min() + h)
        hh = float(Y[m].max() - Y[m].min() + h)
        found.append((round(cy), cx, m, (cx, cy), (w, hh)))
    found.sort(key=lambda f: (f[0], f[1]))
    holes = [Hole(f"H{i + 1}", m, c, s, float(m.sum() * h * h)) for i, (_, _, m, c, s) in enumerate(found)]
    return Grid(h, float(x0), float(y0), nx, ny, domain, holes)


def parse_hole_list(text: str | None, grid: Grid, exclude=()) -> list[str]:
    """'H2-H6', '2,3,5', 'rest' (all holes not in `exclude`) -> hole names."""
    if not text:
        return []
    names = [h.name for h in grid.holes]
    if text.strip().lower() in {"rest", "others", "all-other"}:
        return [n for n in names if n not in exclude]
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.lstrip("Hh")), int(b.lstrip("Hh"))
            out += [f"H{i}" for i in range(a, b + 1)]
        else:
            out.append(f"H{int(part.lstrip('Hh'))}")
    for n in out:
        if n not in names:
            raise KeyError(f"No hole called {n}; holes are {', '.join(names)}")
    return list(dict.fromkeys(out))
