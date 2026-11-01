"""OpenHeart — cardiac MRI analysis SDK.

    import openheart as oh

    pipe = oh.Pipeline([oh.Segment(model), oh.Realign(), oh.FitMesh(), oh.Measure()])
    case = pipe(oh.Case.load("patient001/", layout="legacy"))
    table = pipe.run(oh.Cohort.load("center_A/", layout="legacy"), "out/", jobs=4).to_dataframe()

Heavy dependencies (torch, nnU-Net, VTK) are only imported by the steps that need
them, when they first run.
"""

__version__ = "0.2.0"

from .case import Case, Cohort
from .fit import FitMesh
from .image import Image
from .labels import LabelMap
from .layouts import LAYOUTS, Layout, get_layout
from .measure import Measure
from .pipeline import CaseResult, CohortResult, Pipeline, Step
from .postprocess import FillApex, Realign
from .segment import Segment

__all__ = [
    "Case", "Cohort", "Image", "LabelMap", "Layout", "LAYOUTS", "get_layout",
    "Pipeline", "Step", "CaseResult", "CohortResult",
    "Segment", "Realign", "FillApex", "FitMesh", "Measure",
]
