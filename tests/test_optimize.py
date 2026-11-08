import numpy as np
import pytest
from matplotlib.path import Path

from onshape_topopt.geometry import make_grid, plate_from_tessellation
from onshape_topopt.optimize import Settings, element_stiffness, optimize, ring_mask
from onshape_topopt.profile import ProfileSettings, extract_profile, polygon_area


def test_element_stiffness_is_symmetric_with_rigid_modes():
    KE = element_stiffness(0.3)
    assert np.allclose(KE, KE.T)
    assert np.sum(np.linalg.eigvalsh(KE) < 1e-10) == 3      # 2 translations + 1 rotation


def test_ring_surrounds_hole(bracket):
    grid = make_grid(plate_from_tessellation(bracket), 0.5)
    ring = ring_mask(grid, 2.0)
    assert not (ring & grid.holes[0].mask).any()
    assert ring.sum() > 0 and (ring <= grid.domain).all()


@pytest.fixture(scope="module")
def bracket_run():
    from conftest import box_plate_tessellation

    tess = box_plate_tessellation((0.0, 0.060, 0.0, 0.020),
                                  [(0.004, 0.008, 0.008, 0.012), (0.052, 0.056, 0.008, 0.012)])
    plate = plate_from_tessellation(tess)
    grid = make_grid(plate, 1.0)
    s = Settings(volfrac=0.5, rmin=2.5, ring=2.0, max_iter=60, beta_every=15, beta_max=8)
    result = optimize(grid, ["H1"], ["H2"], s, log=lambda *_: None)
    return plate, grid, result


def test_optimizer_meets_material_budget(bracket_run):
    _, grid, result = bracket_run
    d = result.density[grid.domain]
    assert np.all((d >= 0) & (d <= 1))
    assert result.volume_fraction == pytest.approx(0.5, abs=0.04)
    for v in result.stiffness_retained.values():
        assert 0.2 < v <= 1.0


def test_profile_keeps_holes_inside_material(bracket_run):
    plate, grid, result = bracket_run
    prof = extract_profile(grid, result, plate, ProfileSettings(min_pocket_area=2.0))
    assert not prof.warnings
    outline = Path(prof.outline)
    for hl in grid.holes:
        assert outline.contains_point(hl.centroid)
        assert not any(Path(q).contains_point(hl.centroid) for q in prof.pockets)
    assert polygon_area(prof.outline) > 0                    # counter-clockwise
    frac = prof.stats["final_material_fraction"]
    assert 0.35 < frac < 0.7
    assert prof.stats["min_hole_wall_mm"] is None or prof.stats["min_hole_wall_mm"] > 1.0


def test_rejects_bad_hole_choice(bracket):
    grid = make_grid(plate_from_tessellation(bracket), 1.0)
    with pytest.raises(ValueError):
        optimize(grid, ["H1"], ["H1"])
    with pytest.raises(ValueError):
        optimize(grid, [], ["H2"])
