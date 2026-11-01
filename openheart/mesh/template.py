"""Access to the cardiac template (atlas).

The default template ships inside the package (``assets/template``) so a fresh clone
has everything it needs — no external download. A template provides, per cardiac
phase (ED/ES) and region (LVendo/LVepi/LVmyo/RV), the reference mesh whose topology
is preserved by fitting, plus the template segmentation used to derive template
landmarks for the rigid initialisation. Functions here take ``template``, the folder
of another template with the same files, or ``None`` for the bundled one.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .io.mesh import Mesh
from .io.segmentation import Volume
from .landmarks import extract_landmarks

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(_HERE, "assets", "template")
TEMPLATE_SEG_ED = os.path.join(TEMPLATE_DIR, "segmentation_ED.nii.gz")

REGIONS = ("LVendo", "LVepi", "LVmyo", "RV")
PHASES = ("ED", "ES")


def template_mesh_path(region: str, phase: str, template: str | None = None) -> str:
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; choose from {REGIONS}")
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; choose from {PHASES}")
    return os.path.join(template or TEMPLATE_DIR, "vtks", f"{region}_{phase}.vtk")


@lru_cache(maxsize=None)
def load_template_mesh(region: str, phase: str, template: str | None = None) -> Mesh:
    """Load a template mesh (cached)."""
    path = template_mesh_path(region, phase, template)
    if not os.path.exists(path):
        raise FileNotFoundError(f"template mesh not found: {path}")
    return Mesh.read(path)


@lru_cache(maxsize=None)
def template_landmarks(template: str | None = None):
    """Landmarks of a template, from its own ED segmentation (same routine as subject)."""
    vol = Volume.load(os.path.join(template or TEMPLATE_DIR, "segmentation_ED.nii.gz"))
    return extract_landmarks(vol)
