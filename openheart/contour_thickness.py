"""Myocardial wall thickness from 2D contours, with AHA 17-segment assignment.

Per short-axis slice, the distance from each endocardial contour point to the
nearest epicardial contour point; slices are split into basal / mid / apical
thirds and points assigned to AHA segments relative to the septum.

Adapted from Wenjia Bai's ``cardiac_utils.py`` (ukbb_cardiac, Copyright 2019,
Apache License 2.0). Requires OpenCV (``pip install openheart[contour]``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import interpolate, ndimage
from scipy.spatial import cKDTree

from .labels import CANONICAL

LV, MYO, RV = CANONICAL.lv, CANONICAL.myo, CANONICAL.rv
MIN_PIXELS = 10


def _largest_cc(binary):
    cc, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    areas = ndimage.sum(np.ones_like(cc), cc, index=range(1, n + 1))
    return cc == (int(np.argmax(areas)) + 1)


def _remove_small_cc(binary, thres=10):
    cc, n = ndimage.label(binary)
    out = np.copy(binary)
    for i in range(1, n + 1):
        if np.sum(cc == i) < thres:
            out[cc == i] = 0
    return out


def _smooth_contour(contour, factor=4, smooth=0.05):
    """Periodic spline approximation of a closed contour, upsampled ``factor`` times."""
    n = len(contour)
    dt = 1.0 / n
    t = np.arange(n) * dt
    r = int(0.5 * n)
    t_pad = np.concatenate((np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    x_pad = np.concatenate((contour[-r:, 0], contour[:, 0], contour[:r, 0]))
    y_pad = np.concatenate((contour[-r:, 1], contour[:, 1], contour[:r, 1]))
    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))
    t2 = np.arange(n * factor) / (n * factor)
    return np.stack((fx(t2), fy(t2)), axis=1)


def _outer_contour(mask):
    import cv2

    contours, _ = cv2.findContours(cv2.inRange(mask.astype(np.uint8), 1, 1), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    return contours[0][:, 0, :] if contours else None


def _slice_masks(seg_z):
    endo = _largest_cc(seg_z == LV).astype(np.uint8)
    myo = _remove_small_cc((seg_z == MYO).astype(np.uint8))
    epi = _largest_cc((endo | myo) > 0).astype(np.uint8)
    return endo, myo, epi


def _aha_axes(seg, affine):
    """AHA coordinate system from a mid-cavity slice: LV→septum, inferior→anterior, apex→base."""
    import cv2

    Z = seg.shape[2]
    z = int(Z / 2)
    for off in range(min(5, Z // 2)):
        hit = next((zt for zt in (z + off, z - off) if 0 <= zt < Z
                    and all(np.sum(seg[:, :, zt] == lab) > 50 for lab in (LV, MYO, RV))), None)
        if hit is not None:
            z = hit
            break
    endo, _, epi = _slice_masks(seg[:, :, z])
    rv = _largest_cc(seg[:, :, z] == RV).astype(np.uint8)
    epi_contour = _outer_contour(epi)
    if epi_contour is None or not endo.any():
        return None

    septum = []
    for it in range(1, 5):
        if len(septum) >= 3:
            break
        rv_dilated = cv2.dilate(rv, np.ones((3, 3), dtype=np.uint8), iterations=it)
        septum += [[x, y] for y, x in epi_contour if rv_dilated[x, y] == 1]
    cx, cy = [np.mean(v) for v in np.nonzero(endo)]
    mx, my = septum[int(round(0.5 * len(septum)))] if septum else (cx + 20, cy)

    point_septum = (affine @ np.array([mx, my, z, 1]))[:3]
    point_cavity = (affine @ np.array([cx, cy, z, 1]))[:3]
    lv_to_sep = point_septum - point_cavity
    lv_to_sep /= np.linalg.norm(lv_to_sep)
    apex_to_base = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    if apex_to_base[2] < 0:
        apex_to_base = -apex_to_base
    return {"lv_to_sep": lv_to_sep, "apex_to_base": apex_to_base,
            "inf_to_ant": np.cross(apex_to_base, lv_to_sep)}


def _aha_parts(seg, affine):
    """Label each slice that has LV and myocardium as basal, mid or apical (thirds)."""
    X, Y, Z = seg.shape
    z_pos = [(z, (affine @ np.array([X / 2.0, Y / 2.0, z, 1]))[2]) for z in range(Z)
             if np.sum(seg[:, :, z] == LV) >= MIN_PIXELS and np.sum(seg[:, :, z] == MYO) >= MIN_PIXELS]
    z_pos.sort(key=lambda zp: -zp[1])
    n = len(z_pos)
    i1, i2 = int(math.ceil(n / 3.0)), int(math.ceil(2 * n / 3.0))
    return {z: ("basal" if i < i1 else "mid" if i < i2 else "apical") for i, (z, _) in enumerate(z_pos)}


def _aha_segment(point, lv_centre, axes, part):
    d = point - lv_centre
    d = d - np.dot(d, axes["apex_to_base"]) * axes["apex_to_base"]
    angle = math.degrees(math.atan2(np.dot(d, axes["inf_to_ant"]), np.dot(d, axes["lv_to_sep"]))) % 360
    if part == "basal":
        start, n_segs, width = 1, 6, 60
    elif part == "mid":
        start, n_segs, width = 7, 6, 60
    elif part == "apical":
        start, n_segs, width = 13, 4, 90
    else:
        return 17
    return start + int(((angle + 30) % 360) / width) % n_segs


def wall_thickness(seg: np.ndarray, affine: np.ndarray) -> dict:
    """Wall thickness statistics (mm) of a canonical-label short-axis segmentation.

    Returns ``mean``, ``std``, ``median``, ``min``, ``max``, ``n_points``, and
    ``segments`` (AHA segment id -> mean thickness), plus the raw ``points``
    (world mm), ``thickness`` and ``segment_ids`` arrays.
    """
    axes = _aha_axes(seg, affine)
    if axes is None:
        raise ValueError("no epicardial contour near the mid-ventricle; cannot set up AHA axes")
    parts = _aha_parts(seg, affine)
    points, thickness, seg_ids = [], [], []
    for z in sorted(parts):
        part = parts[z]
        endo, myo, epi = _slice_masks(seg[:, :, z])
        if endo.sum() < MIN_PIXELS or myo.sum() < MIN_PIXELS:
            continue
        cx, cy = [np.mean(v) for v in np.nonzero(endo)]
        lv_centre = (affine @ np.array([cx, cy, z, 1]))[:3]
        endo_c, epi_c = _outer_contour(endo), _outer_contour(epi)
        if endo_c is None or epi_c is None:
            continue
        endo_c, epi_c = _smooth_contour(endo_c), _smooth_contour(epi_c)
        to_world = lambda c: (affine @ np.c_[c[:, 1], c[:, 0], np.full(len(c), z), np.ones(len(c))].T).T[:, :3]  # noqa: E731
        p_endo, p_epi = to_world(endo_c), to_world(epi_c)
        dist, _ = cKDTree(p_epi).query(p_endo)
        points.append(p_endo)
        thickness.append(dist)
        seg_ids += [_aha_segment(p, lv_centre, axes, part) for p in p_endo]
    if not thickness:
        raise ValueError("no slice with both LV cavity and myocardium")
    points, thickness, seg_ids = np.concatenate(points), np.concatenate(thickness), np.array(seg_ids)
    return {
        "mean": float(thickness.mean()), "std": float(thickness.std()),
        "median": float(np.median(thickness)), "min": float(thickness.min()),
        "max": float(thickness.max()), "n_points": int(len(thickness)),
        "segments": {int(s): float(thickness[seg_ids == s].mean()) for s in np.unique(seg_ids)},
        "points": points, "thickness": thickness, "segment_ids": seg_ids,
    }
