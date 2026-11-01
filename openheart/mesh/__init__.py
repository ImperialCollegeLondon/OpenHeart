"""openheart.mesh — GPU cardiac template mesh fitting.

Fit a fixed-topology cardiac template (LV endo/epi/myo + RV) to a segmentation
label map, preserving point correspondence across subjects and phases so the
results can be compared vertex-by-vertex.

Quick start
-----------
>>> from openheart.mesh import fit_subject
>>> results = fit_subject("seg_lvsa_SR_ED.nii.gz", "seg_lvsa_SR_ES.nii.gz",
...                        out_dir="out/")
>>> ed = results["ED"]
>>> ed.meshes["LVepi"].n_points       # same for every subject
12836

This reimplements the mesh-fitting stage of 4DSegment3.0 (originally IRTK
prreg/msrreg/sareg/snreg) in pure Python on the GPU.
"""

from .fitting import FitResult, fit_phase, fit_subject
from .backends.ffd_torch import FFDConfig
from .errors import FitError, InsufficientLandmarksError
from .io.mesh import Mesh
from .io.segmentation import LabelMap, Volume, extract_surface
from .template import REGIONS, PHASES, load_template_mesh

__all__ = [
    "fit_subject",
    "fit_phase",
    "FitResult",
    "FFDConfig",
    "FitError",
    "InsufficientLandmarksError",
    "Mesh",
    "Volume",
    "LabelMap",
    "extract_surface",
    "load_template_mesh",
    "REGIONS",
    "PHASES",
]

