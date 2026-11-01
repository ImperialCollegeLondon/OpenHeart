"""Cardiac measurements: chamber volumes, function and myocardial wall thickness.

Metric names are lower snake case with the unit last, one value per case, e.g.
``lv_edv_ml``, ``lv_ef_pct``, ``lv_mass_g``, ``wt_mean_ed_mm``.
"""

from __future__ import annotations

import numpy as np

from .case import Case
from .image import Image
from .labels import CANONICAL
from .layouts import THICKNESS
from .pipeline import Step

MYOCARDIAL_DENSITY = 1.05  # g/ml


def volumes(seg: Image) -> dict:
    """Blood-pool and myocardial volumes (ml) of one canonical segmentation."""
    voxel_ml = float(np.prod(seg.spacing)) / 1000.0
    return {name: float(np.count_nonzero(seg.array == label)) * voxel_ml
            for name, label in (("lv", CANONICAL.lv), ("myo", CANONICAL.myo), ("rv", CANONICAL.rv))}


def function_metrics(segs: dict) -> dict:
    """Volumes per phase plus stroke volume, ejection fraction and LV mass."""
    v = {phase: volumes(seg) for phase, seg in segs.items()}
    out = {}
    for phase, vol in v.items():
        p = phase.lower()
        out[f"lv_{p}v_ml"] = vol["lv"]
        out[f"rv_{p}v_ml"] = vol["rv"]
        out[f"myo_{p}_ml"] = vol["myo"]
    if "ED" in v:
        out["lv_mass_g"] = v["ED"]["myo"] * MYOCARDIAL_DENSITY
    if "ED" in v and "ES" in v:
        for ch in ("lv", "rv"):
            edv, esv = v["ED"][ch], v["ES"][ch]
            out[f"{ch}_sv_ml"] = edv - esv
            out[f"{ch}_ef_pct"] = 100.0 * (edv - esv) / edv if edv > 0 else float("nan")
    return out


def _summary(values, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_median": float(np.median(values)),
            f"{prefix}_max": float(values.max()), f"{prefix}_p95": float(np.percentile(values, 95))}


class Measure(Step):
    """Compute the per-case metrics row.

    Always: volumes, stroke volume, ejection fraction and LV mass from the
    segmentations. When fitted meshes are present: 3D wall thickness on the LVmyo
    mesh (4DSegment ``cardiacwallthickness``), stored per vertex in
    ``case.features[phase]["LVmyo"]["WallThickness"]`` and summarised as
    ``wt_<stat>_<phase>_mm``.
    With ``contour_thickness=True``: per-slice 2D wall thickness from the
    segmentation with AHA segment means, ``wt2d_*`` (needs OpenCV).
    """

    name = "measure"
    writes = ("features", "metrics")

    def __init__(self, contour_thickness: bool = False):
        self.contour_thickness = contour_thickness

    def process(self, case: Case) -> Case:
        if not case.segs:
            raise ValueError(f"case {case.id} has no segmentation to measure")
        case.metrics.update(function_metrics(case.segs))

        for phase, meshes in case.meshes.items():
            if not {"LVendo", "LVepi", "LVmyo"} <= meshes.keys():
                continue
            from .mesh.features import wall_thickness_myocardium

            myo = wall_thickness_myocardium(meshes["LVendo"], meshes["LVepi"], meshes["LVmyo"])
            thickness = myo.point_scalars[THICKNESS]
            case.features.setdefault(phase, {}).setdefault("LVmyo", {})[THICKNESS] = thickness
            for k, val in _summary(thickness, "wt").items():
                case.metrics[f"{k}_{phase.lower()}_mm"] = val

        if self.contour_thickness:
            from .contour_thickness import wall_thickness

            for phase, seg in case.segs.items():
                p = phase.lower()
                res = wall_thickness(seg.array, seg.affine)
                for k in ("mean", "median", "max"):
                    case.metrics[f"wt2d_{k}_{p}_mm"] = res[k]
                for seg_id, val in res["segments"].items():
                    case.metrics[f"wt2d_aha{seg_id:02d}_{p}_mm"] = val
        return case
