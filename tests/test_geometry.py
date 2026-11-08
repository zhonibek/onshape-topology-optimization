import numpy as np
import pytest
from conftest import box_plate_tessellation, rotate

from onshape_topopt.geometry import NotAPlateError, make_grid, parse_hole_list, plate_from_tessellation


def test_bracket_plate(bracket):
    plate = plate_from_tessellation(bracket)
    assert np.allclose(plate.normal, [0, 0, 1])
    assert plate.thickness_mm == pytest.approx(6.0)
    assert plate.area_mm2() == pytest.approx(60 * 20 - 2 * 16)
    grid = make_grid(plate, 0.5)
    assert [h.name for h in grid.holes] == ["H1", "H2"]
    assert grid.holes[0].centroid == pytest.approx((6.0, 10.0), abs=0.3)
    assert grid.holes[1].centroid == pytest.approx((54.0, 10.0), abs=0.3)
    assert grid.holes[0].area_mm2 == pytest.approx(16, rel=0.15)


def test_world_round_trip(bracket):
    plate = plate_from_tessellation(bracket)
    w = plate.to_world(np.array([[6.0, 10.0]]))
    assert np.allclose(w, [[0.006, 0.010, 0.0]])


def test_tilted_plate_is_still_a_plate(bracket):
    c, s = np.cos(0.3), np.sin(0.3)
    R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    plate = plate_from_tessellation(rotate(bracket, R))
    assert plate.thickness_mm == pytest.approx(6.0)
    assert plate.area_mm2() == pytest.approx(1168, rel=1e-6)
    assert len(make_grid(plate, 0.5).holes) == 2


def test_sloped_face_is_rejected(bracket):
    bracket[0]["faces"].append({"id": "chamfer", "facets": [
        {"normal": [0, 0.7071, 0.7071], "vertices": [[0, 0, 0], [0.01, 0, 0], [0.01, -0.001, -0.001]]}]})
    with pytest.raises(NotAPlateError):
        plate_from_tessellation(bracket)


def test_step_is_rejected():
    a = box_plate_tessellation((0, 0.02, 0, 0.02), [], thickness=0.006, body_id="A")
    b = box_plate_tessellation((0.02, 0.04, 0, 0.02), [], thickness=0.006, z_top=0.002, body_id="B")
    with pytest.raises(NotAPlateError):
        plate_from_tessellation(a + b)


def test_hole_lists(bracket):
    grid = make_grid(plate_from_tessellation(bracket), 0.5)
    assert parse_hole_list("H1", grid) == ["H1"]
    assert parse_hole_list("1-2", grid) == ["H1", "H2"]
    assert parse_hole_list("rest", grid, exclude=["H1"]) == ["H2"]
    with pytest.raises(KeyError):
        parse_hole_list("H7", grid)


def test_intake_side_fixture(intake_side):
    plate = plate_from_tessellation(intake_side)
    assert plate.thickness_mm == pytest.approx(12.7, abs=1e-3)
    # L-shaped plate minus 6 square holes, from the original Onshape parts
    assert plate.area_mm2() == pytest.approx(7188.1, abs=0.5)
    grid = make_grid(plate, 0.5)
    assert len(grid.holes) == 6
    h1 = grid.holes[0]                      # Part 2's single hole sorts first
    assert h1.centroid == pytest.approx((-44.2, 57.15), abs=0.3)
    assert all(c[1] > 165 for c in (h.centroid for h in grid.holes[1:]))
