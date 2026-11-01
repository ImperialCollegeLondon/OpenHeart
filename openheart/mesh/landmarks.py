"""Anatomical landmark extraction from a segmentation.

Pure-numpy port of ``mesh_ed_es.py::ExtractLandmarks._extract_landmarks`` from
4DSegment3.0. Landmarks are basal / mid / apical centroids of each chamber label,
used to initialise the rigid alignment between subject and template (the analogue
of the IRTK ``prreg`` point-registration step).
"""

from __future__ import annotations

import numpy as np

from .io.segmentation import LabelMap, Volume


def extract_landmarks(
    volume: Volume,
    labels=None,
    *,
    mirror_xaxis: bool = False,
) -> np.ndarray:
    """Extract 3-per-chamber landmarks in world coordinates.

    Faithful port of the original logic: for each of the LV cavity, myocardium and
    RV, take base/mid/apex slices along the header z-axis and use the in-slice
    centroid as the landmark. The unreliable top landmark of the LV cavity and of
    the RV is dropped, exactly as upstream, yielding 7 landmarks for the default
    three-label convention.

    Parameters
    ----------
    labels : the segmentation's label convention — a ``LabelMap``, an
        ``'lv=..,myo=..,rv=..'`` string, or ``None``. When
        ``None`` the convention the ``volume`` was loaded with (``volume.labels``)
        is used, so callers normally don't pass this explicitly.

    Returns
    -------
    (K, 3) float array of world-coordinate landmark points.
    """
    lmap = (
        LabelMap.from_spec(labels)
        if labels is not None
        else getattr(volume, "labels", None) or LabelMap()
    )

    seg = np.squeeze(volume.data)
    affine = volume.affine

    # Determine which z end is the BASE from anatomy, not from the affine z-column
    # sign. The original 4DSegment used ``affine[:3,2][2] < 0`` to pick the base
    # end, but that assumes the affine faithfully encodes slice orientation. Some
    # exports have a degenerate/rebuilt affine whose z sign does NOT match the
    # true slice order (base actually sits at the low-z end while the affine says
    # otherwise), so the heuristic mis-orders the landmarks and the rigid init comes
    # out flipped. The base of the ventricle has the LARGER blood-pool cross-section,
    # so we compare the LV-cavity area at the two z extremes directly.
    lv_z = np.nonzero(seg == lmap.lv)[2]
    if lv_z.size:
        lv_zmin, lv_zmax = int(lv_z.min()), int(lv_z.max())
        area_lo = int((seg[:, :, lv_zmin] == lmap.lv).sum())
        area_hi = int((seg[:, :, lv_zmax] == lmap.lv).sum())
        base_at_zmin = area_lo >= area_hi  # larger cavity slice == base
    else:
        base_at_zmin = True

    # RV may carry an extra label (e.g. 4); use whichever RV label is present.
    rv_candidates = [l for l in ((lmap.rv,) + tuple(lmap.rv_extra)) if np.any(seg == l)]
    rv_label = max(rv_candidates) if rv_candidates else lmap.rv

    # (label, drop_top_landmark) in the original LV-cavity, myo, RV order. The top
    # (basal) landmark of the LV cavity and the RV is dropped as unreliable.
    plan = [(lmap.lv, True), (lmap.myo, False), (rv_label, True)]

    lm = []
    for lbl, drop_top in plan:
        z = np.nonzero(seg == lbl)[2]
        if z.size == 0:
            z_min = z_max = 0
        else:
            z_min, z_max = int(z.min()), int(z.max())
        z_mid = int(round(0.5 * (z_min + z_max)))

        # Order the slices base -> mid -> apex (so dropping the unreliable basal
        # landmark below removes the right one).
        if base_at_zmin:
            zs = [z_min, z_mid, z_max]
        else:
            zs = [z_max, z_mid, z_min]

        if drop_top:  # drop the (unreliable) basal landmark for LV cavity / RV
            zs = zs[1:]

        for z in zs:
            sl = seg[:, :, z] == lbl
            if not sl.any():
                continue
            x, y = (np.mean(i) for i in np.nonzero(sl))
            p = affine @ np.array([x, y, z, 1.0]).reshape(4, 1)
            p = p[:3, 0]
            if mirror_xaxis:
                p[0] = -p[0]
            lm.append(p)

    return np.asarray(lm, dtype=np.float64)


def empty_chambers(volume: Volume, labels=None) -> tuple:
    """Names of the chambers (``LV`` / ``myo`` / ``RV``) that are empty in ``volume``.

    Used to explain *why* landmark extraction came up short (see
    :class:`~openheart.mesh.errors.InsufficientLandmarksError`).
    """
    lmap = (
        LabelMap.from_spec(labels)
        if labels is not None
        else getattr(volume, "labels", None) or LabelMap()
    )
    seg = np.squeeze(volume.data)
    rv_labels = (lmap.rv,) + tuple(lmap.rv_extra)
    missing = []
    if not np.any(seg == lmap.lv):
        missing.append("LV")
    if not np.any(seg == lmap.myo):
        missing.append("myo")
    if not any(np.any(seg == l) for l in rv_labels):
        missing.append("RV")
    return tuple(missing)
