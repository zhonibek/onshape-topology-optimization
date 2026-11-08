import numpy as np
import pytest

from onshape_topopt.cad import SketchPlane, extrude_feature, pick_plane, sketch_feature, union_feature
from onshape_topopt.geometry import plate_from_tessellation
from onshape_topopt.onshape import detect_style, fs_value, parse_url, serialize


def test_parse_url():
    ref = parse_url("https://cad.onshape.com/documents/0123456789abcdef01234567/w/89abcdef0123456789abcdef"
                    "/e/fedcba9876543210fedcba98")
    assert ref.base_url == "https://cad.onshape.com"
    assert ref.path == "d/0123456789abcdef01234567/w/89abcdef0123456789abcdef/e/fedcba9876543210fedcba98"
    assert ref.writable
    with pytest.raises(ValueError):
        parse_url("https://example.com/nothing")


def test_two_json_styles():
    f = union_feature("g", ["JqD", "JqH"])
    msg = serialize(f, "message")
    assert msg["typeName"] == "BTMFeature" and msg["type"] == 134
    q = msg["message"]["parameters"][1]["message"]["queries"][0]
    assert q["message"]["geometryIds"] == ["JqD", "JqH"]
    bt = serialize(f, "btType")
    assert bt["btType"] == "BTMFeature-134"
    assert bt["parameters"][1]["queries"][0]["deterministicIds"] == ["JqD", "JqH"]
    assert detect_style({"features": [msg]}) == "message"
    assert detect_style({"features": [bt]}) == "btType"


def test_fs_value_decodes_both_styles():
    msg = {"type": 1499, "typeName": "BTFSValueArray", "message": {"value": [
        {"type": 1422, "typeName": "BTFSValueString", "message": {"value": "JDC"}},
        {"type": 772, "typeName": "BTFSValueNumber", "message": {"value": 1}}]}}
    bt = {"btType": "BTFSValueArray-1499", "value": [{"btType": "BTFSValueString-1422", "value": "JDC"},
                                                      {"btType": "BTFSValueNumber-772", "value": 1}]}
    assert fs_value(msg) == ["JDC", 1] == fs_value(bt)


def test_sketch_lines_close_the_loop():
    square = np.array([[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]])
    sk = serialize(sketch_feature("s", "JDC", [square], "o"), "message")["message"]
    ents = [e["message"] for e in sk["entities"]]
    assert len(ents) == 4
    for e, (a, b) in zip(ents, zip(square, np.roll(square, -1, axis=0))):
        g = e["geometry"]["message"]
        end = np.array([g["pntX"], g["pntY"]]) + e["endParam"] * np.array([g["dirX"], g["dirY"]])
        assert np.allclose([g["pntX"], g["pntY"]], a) and np.allclose(end, b)


def test_extrude_options():
    through = serialize(extrude_feature("e", "F1", ["JuC"], "INTERSECT", "JqD", opposite=True), "message")["message"]
    p = {x["message"]["parameterId"]: x["message"] for x in through["parameters"]}
    assert p["operationType"]["value"] == "INTERSECT"
    assert p["endBound"]["value"] == "THROUGH_ALL" and p["oppositeDirection"]["value"] is True
    assert "hasSecondDirection" not in p     # two through-all directions failed in Onshape
    # only the plate is cut, never other parts in the Part Studio
    assert p["defaultScope"]["value"] is False
    assert p["booleanScope"]["queries"][0]["message"]["geometryIds"] == ["JqD"]
    sym = serialize(extrude_feature("e", "F1", ["JuC"], "REMOVE", "JqD", symmetric_depth_mm=30), "message")["message"]
    p = {x["message"]["parameterId"]: x["message"] for x in sym["parameters"]}
    assert p["symmetric"]["value"] is True and p["depth"]["expression"] == "30.000 mm"


def _top(z=0.0):
    return SketchPlane("Top", "JDC", np.array([0, 0, z]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))


def test_pick_plane_sides(bracket):
    plate = plate_from_tessellation(bracket)                  # z from -6 mm to 0
    assert pick_plane([_top(0.0)], plate)[1] == {"opposite": True}
    assert pick_plane([_top(-0.006)], plate)[1] == {"opposite": False}
    how = pick_plane([_top(-0.003)], plate)[1]
    assert how["symmetric_depth_mm"] == pytest.approx(8.0)


def test_sketch_coordinates_on_top_plane(bracket):
    plate = plate_from_tessellation(bracket)
    pl = _top()
    pts = pl.to_sketch(plate.to_world(np.array([[6.0, 10.0], [54.0, 10.0]])))
    assert np.allclose(pts, [[0.006, 0.010], [0.054, 0.010]])
