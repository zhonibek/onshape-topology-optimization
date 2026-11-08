import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def box_plate_tessellation(outer, holes, thickness=0.006, z_top=0.0, body_id="B1", sides=True):
    """Onshape-style tessellation (metres) of a rectangular plate with rectangular holes.

    outer / holes are (xmin, xmax, ymin, ymax) in metres; the top face is at z_top.
    """
    xs = sorted({outer[0], outer[1], *[h[0] for h in holes], *[h[1] for h in holes]})
    ys = sorted({outer[2], outer[3], *[h[2] for h in holes], *[h[3] for h in holes]})
    tris = []
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(h[0] < cx < h[1] and h[2] < cy < h[3] for h in holes):
                continue
            tris += [[(x0, y0), (x1, y0), (x1, y1)], [(x0, y0), (x1, y1), (x0, y1)]]
    zb = z_top - thickness
    top = [{"normal": [0, 0, 1], "vertices": [[x, y, z_top] for x, y in t]} for t in tris]
    bot = [{"normal": [0, 0, -1], "vertices": [[x, y, zb] for x, y in t[::-1]]} for t in tris]
    faces = [{"id": "top", "facets": top}, {"id": "bottom", "facets": bot}]
    if sides:  # one side wall, enough to check the "perpendicular faces are fine" path
        x0, x1, y0 = outer[0], outer[1], outer[2]
        faces.append({"id": "side", "facets": [
            {"normal": [0, -1, 0], "vertices": [[x0, y0, zb], [x1, y0, zb], [x1, y0, z_top]]}]})
    return [{"id": body_id, "bodyType": "solid", "faces": faces}]


@pytest.fixture
def bracket():
    """60 x 20 mm plate, 6 mm thick, one 4 mm hole near each end."""
    return box_plate_tessellation(
        (0.0, 0.060, 0.0, 0.020),
        [(0.004, 0.008, 0.008, 0.012), (0.052, 0.056, 0.008, 0.012)],
    )


@pytest.fixture
def intake_side():
    data = json.loads((ROOT / "examples" / "intake-side" / "tessellation.json").read_text())
    return data["bodies"]


def rotate(tess, R):
    out = json.loads(json.dumps(tess))
    for b in out:
        for f in b["faces"]:
            for ft in f["facets"]:
                ft["normal"] = list(np.asarray(R) @ ft["normal"])
                ft["vertices"] = [list(np.asarray(R) @ v) for v in ft["vertices"]]
    return out
