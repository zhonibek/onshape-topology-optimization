"""Topology optimization for flat plates designed in Onshape."""

__version__ = "0.1.0"

from .geometry import Grid, Hole, NotAPlateError, Plate, make_grid, parse_hole_list, plate_from_tessellation
from .optimize import Result, Settings, optimize
from .profile import Profile, ProfileSettings, extract_profile

__all__ = [
    "Grid", "Hole", "NotAPlateError", "Plate", "make_grid", "parse_hole_list", "plate_from_tessellation",
    "Result", "Settings", "optimize", "Profile", "ProfileSettings", "extract_profile", "__version__",
]
