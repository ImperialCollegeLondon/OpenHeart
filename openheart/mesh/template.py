"""Access to the bundled cardiac template (atlas).

The template ships inside the package (``assets/template``) so a fresh clone has
everything it needs — no external download. It provides, per cardiac phase
(ED/ES) and region (LVendo/LVepi/LVmyo/RV), the reference mesh whose topology is
preserved by fitting, plus the template segmentation used to derive template
landmarks for the rigid initialisation.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .io.mesh import Mesh
from .io.segmentation import Volume
from .landmarks import extract_landmarks

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(_HERE, "assets", "template")
TEMPLATE_VTK_DIR = os.path.join(TEMPLATE_DIR, "vtks")
TEMPLATE_SEG_ED = os.path.join(TEMPLATE_DIR, "segmentation_ED.nii.gz")

REGIONS = ("LVendo", "LVepi", "LVmyo", "RV")
PHASES = ("ED", "ES")


def template_mesh_path(region: str, phase: str) -> str:
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; choose from {REGIONS}")
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; choose from {PHASES}")
    return os.path.join(TEMPLATE_VTK_DIR, f"{region}_{phase}.vtk")


@lru_cache(maxsize=None)
def load_template_mesh(region: str, phase: str) -> Mesh:
    """Load a bundled template mesh (cached)."""
    return Mesh.read(template_mesh_path(region, phase))


@lru_cache(maxsize=1)
def template_landmarks():
    """Landmarks of the template, from its own ED segmentation (same routine as subject)."""
    vol = Volume.load(TEMPLATE_SEG_ED)
    return extract_landmarks(vol)
