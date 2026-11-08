"""Write an optimized profile back into an Onshape Part Studio as editable features.

Recipe (each step is checked, and everything added is rolled back on failure):
  1. Boolean union of the selected parts (when there is more than one)
  2. Sketch of the outline on the default plane parallel to the plate
  3. Extrude "Intersect" with that sketch  -> trims the outer shape
  4. Sketch of the pockets
  5. Extrude "Remove" with that sketch     -> cuts the pockets
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Grid, Plate
from .onshape import Client, ElementRef, Node, OnshapeError
from .profile import Profile, polygon_area


# ---------------------------------------------------------------- feature builders
def _query_list(pid: str, queries: list) -> Node:
    return Node(148, "BTMParameterQueryList", parameterId=pid, queries=queries)


def _enum(pid: str, enum: str, value: str) -> Node:
    return Node(145, "BTMParameterEnum", parameterId=pid, enumName=enum, value=value)


def _bool(pid: str, value: bool) -> Node:
    return Node(144, "BTMParameterBoolean", parameterId=pid, value=value)


def _quantity(pid: str, expression: str) -> Node:
    return Node(147, "BTMParameterQuantity", parameterId=pid, expression=expression)


def _ids(ids: list[str]) -> dict:
    # written as geometryIds ("message" style) or deterministicIds ("btType" style)
    return {"geometryIds": list(ids)}


def union_feature(name: str, part_ids: list[str]) -> Node:
    return Node(134, "BTMFeature", featureType="booleanBodies", name=name, parameters=[
        _enum("operationType", "BooleanOperationType", "UNION"),
        _query_list("tools", [Node(138, "BTMIndividualQuery", **_ids(part_ids))]),
    ])


def sketch_feature(name: str, plane_id: str, loops_m: list[np.ndarray], prefix: str) -> Node:
    """Closed polylines (sketch-plane coordinates, metres) as line segments."""
    ents = []
    for li, loop in enumerate(loops_m):
        for k in range(len(loop)):
            a, b = loop[k], loop[(k + 1) % len(loop)]
            L = float(np.hypot(*(b - a)))
            if L < 1e-7:
                continue
            eid = f"{prefix}{li}_{k}"
            ents.append(Node(155, "BTMSketchCurveSegment", entityId=eid, startPointId=f"{eid}.start",
                             endPointId=f"{eid}.end", startParam=0.0, endParam=L, isConstruction=False,
                             geometry=Node(117, "BTCurveGeometryLine", pntX=float(a[0]), pntY=float(a[1]),
                                           dirX=float((b[0] - a[0]) / L), dirY=float((b[1] - a[1]) / L))))
    return Node(151, "BTMSketch", featureType="newSketch", name=name, entities=ents, constraints=[], parameters=[
        _query_list("sketchPlane", [Node(138, "BTMIndividualQuery", **_ids([plane_id]))]),
        _bool("disableImprinting", True),
    ])


def extrude_feature(name: str, sketch_id: str, region_ids: list[str], operation: str, part_id: str,
                    opposite: bool = False, symmetric_depth_mm: float | None = None) -> Node:
    """Extrude that only acts on `part_id` (other parts in the Part Studio are left alone)."""
    params = [
        _enum("bodyType", "ExtendedToolBodyType", "SOLID"),
        _enum("operationType", "NewBodyOperationType", operation),
        _query_list("entities", [Node(140, "BTMIndividualSketchRegionQuery", featureId=sketch_id,
                                      filterInnerLoops=True, **_ids(region_ids))]),
        _bool("defaultScope", False),
        _query_list("booleanScope", [Node(138, "BTMIndividualQuery", **_ids([part_id]))]),
    ]
    if symmetric_depth_mm is None:
        params += [_enum("endBound", "BoundingType", "THROUGH_ALL"), _bool("oppositeDirection", opposite)]
    else:
        params += [_enum("endBound", "BoundingType", "BLIND"), _bool("symmetric", True),
                   _quantity("depth", f"{symmetric_depth_mm:.3f} mm")]
    return Node(134, "BTMFeature", featureType="extrude", name=name, parameters=params)


# ---------------------------------------------------------------- sketch plane
@dataclass
class SketchPlane:
    name: str
    id: str
    origin: np.ndarray
    x: np.ndarray
    normal: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return np.cross(self.normal, self.x)

    def to_sketch(self, world: np.ndarray) -> np.ndarray:
        rel = np.asarray(world) - self.origin
        return np.stack([rel @ self.x, rel @ self.y], axis=-1)


PLANES_SCRIPT = """function(context is Context, queries) {
  var out = [];
  for (var name in ["Top", "Front", "Right"]) {
    var q = qCreatedBy(makeId(name), EntityType.FACE);
    var p = evPlane(context, {"face" : q});
    out = append(out, [name, transientQueriesToStrings(evaluateQuery(context, q))[0],
      p.origin[0] / meter, p.origin[1] / meter, p.origin[2] / meter,
      p.x[0], p.x[1], p.x[2], p.normal[0], p.normal[1], p.normal[2]]);
  }
  return out;
}"""


def default_planes(client: Client, ref: ElementRef) -> list[SketchPlane]:
    rows = client.eval_featurescript(ref, PLANES_SCRIPT)
    return [SketchPlane(r[0], r[1], np.array(r[2:5], float), np.array(r[5:8], float), np.array(r[8:11], float))
            for r in rows]


def pick_plane(planes: list[SketchPlane], plate: Plate):
    """Default plane parallel to the plate, and how to extrude through the plate from it."""
    for pl in planes:
        if abs(pl.normal @ plate.normal) > 0.9999:
            s = [float((plate.normal * hgt - pl.origin) @ pl.normal) for hgt in (plate.top, plate.bottom)]
            if max(s) <= 1e-7:
                return pl, {"opposite": True}
            if min(s) >= -1e-7:
                return pl, {"opposite": False}
            return pl, {"symmetric_depth_mm": 2000 * max(abs(v) for v in s) + 2.0}
    raise OnshapeError("The plate is not parallel to the Top, Front or Right plane; this is not supported yet.")


# ---------------------------------------------------------------- apply
def _region_ids(client, ref, sketch_id):
    script = (f'function(context is Context, queries) {{ var q = qSketchRegion(makeId("{sketch_id}")); '
              f'return [transientQueriesToStrings(evaluateQuery(context, q)), '
              f'evArea(context, {{"entities" : q}}) / (meter * meter)]; }}')
    ids, area = client.eval_featurescript(ref, script)
    return list(ids), float(area)


def _holes_open(client, ref, plate: Plate, grid: Grid) -> list[str]:
    """Names of holes whose centre is no longer empty, or whose wall is gone."""
    mid = (plate.top + plate.bottom) / 2
    checks, names = [], []
    for hl in grid.holes:
        cx, cy = hl.centroid
        for pt, want in (((cx, cy), False), ((cx + hl.size[0] / 2 + 1.0, cy), True)):
            w = plate.to_world(np.array(pt), height=mid) * 1000
            checks.append(f'!isQueryEmpty(context, qContainsPoint(qBodyType(qEverything(EntityType.BODY), '
                          f'BodyType.SOLID), vector({w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f}) * millimeter)) == {str(want).lower()}')
            names.append(hl.name)
    ok = client.eval_featurescript(ref, f"function(context is Context, queries) {{ return [{', '.join(checks)}]; }}")
    return sorted({n for n, good in zip(names, ok) if not good})


def apply_profile(client: Client, ref: ElementRef, plate: Plate, grid: Grid, profile: Profile,
                  part_ids: list[str], prefix: str = "TopOpt", log=print) -> dict:
    if not ref.writable:
        raise OnshapeError("Open the document's workspace (URL with /w/) to write features.")
    added: list[str] = []
    t_m = (plate.top - plate.bottom)

    def add(feature: Node, what: str) -> str:
        res = client.add_feature(ref, feature)
        if res["featureId"]:
            added.append(res["featureId"])
        if res["status"] != "OK":
            raise OnshapeError(f"{what} failed in Onshape (status {res['status']}).")
        log(f"  added {what} ({res['featureId']})")
        return res["featureId"]

    def total_volume_mm3() -> float:
        return sum(client.volumes(ref).values()) * 1e9

    try:
        if len(part_ids) > 1:
            add(union_feature(f"{prefix} - group parts", part_ids), "Boolean union")
        left = [p["partId"] for p in client.parts(ref) if p.get("partId") in part_ids]
        if len(left) != 1:
            raise OnshapeError(f"expected one merged part, found {len(left)} (do the parts touch?)")
        target = left[0]
        before = total_volume_mm3()

        plane, how = pick_plane(default_planes(client, ref), plate)
        log(f"  sketching on the {plane.name} plane")

        def to_sketch(loop_mm):
            return plane.to_sketch(plate.to_world(loop_mm))

        for label, loops, op, expected_area, op_name in (
            ("outline", [profile.outline], "INTERSECT", abs(polygon_area(profile.outline)), "trim outline"),
            ("pockets", profile.pockets, "REMOVE", sum(abs(polygon_area(q)) for q in profile.pockets), "cut pockets"),
        ):
            if not loops:
                continue
            sk = add(sketch_feature(f"{prefix} - {label}", plane.id, [to_sketch(lp) for lp in loops], label[0]),
                     f"{label} sketch")
            ids, area_m2 = _region_ids(client, ref, sk)
            if len(ids) != len(loops) or abs(area_m2 * 1e6 - expected_area) > 0.005 * expected_area + 0.5:
                raise OnshapeError(f"{label} sketch has {len(ids)} regions / {area_m2 * 1e6:.1f} mm^2, "
                                   f"expected {len(loops)} / {expected_area:.1f} mm^2")
            add(extrude_feature(f"{prefix} - {op_name}", sk, ids, op, target, **how), op_name)

        # compare the volume removed (other parts in the studio cancel out)
        after = total_volume_mm3()
        removed = before - after
        expected = (profile.stats["plate_area_mm2"] - profile.stats["final_area_mm2"]) * t_m * 1000
        if abs(removed - expected) > 0.03 * expected + 1.0:
            raise OnshapeError(f"removed {removed:.0f} mm^3 of material, expected about {expected:.0f} mm^3")
        bad = _holes_open(client, ref, plate, grid)
        if bad:
            raise OnshapeError(f"screw holes damaged: {', '.join(bad)} "
                               "(a modelled screw or another part inside a hole also triggers this)")
    except Exception:
        log("  something did not match; removing the features this run added")
        for fid in reversed(added):
            try:
                client.delete_feature(ref, fid)
            except OnshapeError as e:  # keep going, report at the end
                log(f"  could not delete {fid}: {e}")
        raise
    plate_volume = profile.stats["plate_area_mm2"] * t_m * 1000
    return {"features": added, "part_id": target, "volume_before_mm3": plate_volume,
            "volume_after_mm3": plate_volume - removed, "material_kept": 1 - removed / plate_volume}
