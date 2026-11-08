"""Command line: onshape-topopt holes|run <part-studio-url> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .geometry import make_grid, parse_hole_list, plate_from_tessellation
from .optimize import LOAD_CASES, Settings, optimize
from .profile import ProfileSettings, extract_profile
from .report import plot_holes, plot_result


def _source(args):
    """Returns (client, ref, tessellation, part_ids, part_names)."""
    if args.tessellation:
        data = json.loads(Path(args.tessellation).read_text())
        bodies = data["bodies"] if isinstance(data, dict) else data
        names = data.get("parts", {}) if isinstance(data, dict) else {}
        ids = [b["id"] for b in bodies]
        names = {i: names.get(i, i) for i in ids}
        ids = _select(ids, names, args.parts)
        return None, None, bodies, ids, names
    if not args.url:
        sys.exit("Give a Part Studio URL or --tessellation FILE.")
    from .onshape import Client, parse_url

    client, ref = Client(), parse_url(args.url)
    parts = [p for p in client.parts(ref) if p.get("bodyType", "solid") == "solid"]
    names = {p["partId"]: p["name"] for p in parts}
    ids = _select(list(names), names, args.parts)
    tess = client.tessellation(ref, ids)
    if getattr(args, "save_tessellation", None):
        Path(args.save_tessellation).write_text(json.dumps({"source": args.url, "parts": names, "bodies": tess}))
    return client, ref, tess, ids, names


def _select(ids, names, wanted):
    if not wanted:
        return ids
    by_name = {n.lower(): i for i, n in names.items()}
    out = []
    for w in (s.strip() for s in wanted.split(",")):
        key = w.lower()
        if key in by_name:
            out.append(by_name[key])
        elif w in ids:
            out.append(w)
        else:
            sys.exit(f"No part called '{w}'. Parts: {', '.join(names.values())}")
    return out


def _print_holes(grid):
    print(f"{'hole':5} {'centre u, v (mm)':>20} {'size (mm)':>14}")
    for hl in grid.holes:
        print(f"{hl.name:5} {hl.centroid[0]:9.2f}, {hl.centroid[1]:8.2f} {hl.size[0]:6.1f} x {hl.size[1]:4.1f}")


def cmd_holes(args):
    _, _, tess, ids, names = _source(args)
    plate = plate_from_tessellation(tess, ids)
    grid = make_grid(plate, args.element)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"plate from {', '.join(names[i] for i in ids)}: {plate.thickness_mm:.2f} mm thick, "
          f"{plate.area_mm2():.0f} mm^2, {len(grid.holes)} holes")
    _print_holes(grid)
    plot_holes(grid, out / "holes.png")
    print(f"labelled picture: {out / 'holes.png'}")


def cmd_run(args):
    client, ref, tess, ids, names = _source(args)
    plate = plate_from_tessellation(tess, ids)
    grid = make_grid(plate, args.element)
    fixed = parse_hole_list(args.fixed, grid)
    loaded = parse_hole_list(args.load or "rest", grid, exclude=fixed)
    print(f"plate: {plate.thickness_mm:.2f} mm thick, {plate.area_mm2():.0f} mm^2, {len(grid.holes)} holes; "
          f"fixed {', '.join(fixed)}; loaded {', '.join(loaded)}")
    n_el = int(grid.domain.sum())
    if n_el > 150_000:
        print(f"warning: {n_el} elements; this will be slow. Try a larger --element.")

    settings = Settings(volfrac=args.volfrac, rmin=args.filter, ring=args.ring, max_iter=args.max_iter,
                        load_cases=tuple(c.strip() for c in args.load_cases.split(",")))
    result = optimize(grid, fixed, loaded, settings)
    profile = extract_profile(grid, result, plate, ProfileSettings(min_pocket_area=args.min_pocket,
                                                                    pocket_smoothing=args.pocket_smoothing))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_result(grid, result, profile, out / "preview.png", fixed, loaded)
    st = profile.stats
    summary = {
        "version": __version__,
        "parts": [names[i] for i in ids],
        "thickness_mm": plate.thickness_mm,
        "fixed_holes": fixed,
        "loaded_holes": loaded,
        "settings": vars(settings) | {"element_mm": args.element},
        "iterations": result.iterations,
        "runtime_s": round(result.runtime_s, 1),
        "material_fraction_grid": result.volume_fraction,
        "stiffness_retained": result.stiffness_retained,
        **st,
        "warnings": profile.warnings,
    }
    frame = {"normal": plate.normal.tolist(), "u": plate.u.tolist(), "v": plate.v.tolist(),
             "top_m": plate.top, "bottom_m": plate.bottom}
    (out / "result.json").write_text(json.dumps({
        "summary": summary, "plate_frame": frame,
        "outline_mm": profile.outline.round(3).tolist(),
        "pockets_mm": [q.round(3).tolist() for q in profile.pockets],
    }, indent=1))

    print(f"\nkeeps {st['final_material_fraction']:.1%} of the material; stiffness kept: "
          + ", ".join(f"{k} {v:.0%}" for k, v in result.stiffness_retained.items()))
    print(f"{st['pockets']} pockets, thinnest member {st['thinnest_member_mm']:.2f} mm, "
          f"narrowest pocket {st['narrowest_pocket_mm']:.2f} mm, wall around holes >= {st['min_hole_wall_mm']:.2f} mm"
          if st["pockets"] else "no pockets")
    for w in profile.warnings:
        print("warning:", w)
    print(f"preview: {out / 'preview.png'}\nresult:  {out / 'result.json'}")

    if args.apply:
        if client is None:
            sys.exit("--apply needs a Part Studio URL (not --tessellation).")
        if profile.warnings and not args.force:
            sys.exit("Not writing to Onshape because of the warnings above (use --force to write anyway).")
        from .cad import apply_profile

        print("\nwriting features to Onshape ...")
        res = apply_profile(client, ref, plate, grid, profile, ids, prefix=args.prefix)
        print(f"done: volume {res['volume_before_mm3']:.0f} -> {res['volume_after_mm3']:.0f} mm^3 "
              f"({res['material_kept']:.1%} kept)")
    else:
        print("\n(dry run: nothing written to Onshape; add --apply to cut the part)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="onshape-topopt", description="Topology optimization for flat Onshape plates.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("url", nargs="?", help="Part Studio URL (https://cad.onshape.com/documents/...)")
        sp.add_argument("--tessellation", help="use a saved tessellation JSON instead of Onshape")
        sp.add_argument("--parts", help="comma-separated part names (default: every solid part)")
        sp.add_argument("--element", type=float, default=0.5, help="element size in mm (default 0.5)")
        sp.add_argument("--out", default="topopt-output", help="output folder")

    sh = sub.add_parser("holes", help="list and label the holes of the plate")
    common(sh)
    sh.set_defaults(func=cmd_holes)

    sr = sub.add_parser("run", help="optimize the plate (and optionally write the cuts to Onshape)")
    common(sr)
    sr.add_argument("--fixed", required=True, help="holes held still, e.g. H1 or 1,2")
    sr.add_argument("--load", help="holes where force comes in, e.g. H2-H6 (default: every other hole)")
    sr.add_argument("--volfrac", type=float, default=0.40, help="share of material to keep (default 0.40)")
    sr.add_argument("--filter", type=float, default=4.5, help="filter radius in mm; bigger = thicker members")
    sr.add_argument("--ring", type=float, default=3.0, help="solid wall kept around each hole in mm")
    sr.add_argument("--load-cases", default=",".join(LOAD_CASES), help="any of x,y,twist (default all)")
    sr.add_argument("--max-iter", type=int, default=260)
    sr.add_argument("--min-pocket", type=float, default=8.0, help="drop pockets smaller than this many mm^2")
    sr.add_argument("--pocket-smoothing", type=float, default=0.8, help="pocket edge smoothing in mm (0 = off)")
    sr.add_argument("--save-tessellation", help="also save the downloaded geometry to this JSON file")
    sr.add_argument("--apply", action="store_true", help="write the result into the Part Studio")
    sr.add_argument("--force", action="store_true", help="write even if there are warnings")
    sr.add_argument("--prefix", default="TopOpt", help="name prefix for the new Onshape features")
    sr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
