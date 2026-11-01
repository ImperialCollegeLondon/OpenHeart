"""Segmentation post-processing: slice realignment and LV apex completion.

Both functions take canonical-label segmentations (1=LV, 2=myo, 3=RV) and work
slice by slice along the short-axis stack (array axis k).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, shift

from .case import Case
from .image import Image
from .labels import CANONICAL
from .pipeline import Step


class RealignError(ValueError):
    """Raised when a stack has too few segmented slices to fit a centreline."""


def realign(image: Image, seg: Image, degree: int = 1):
    """Straighten a stack by shifting each slice onto a fitted LV/RV centreline.

    Fits a polynomial of ``degree`` through the per-slice centroids of the
    segmentation and translates each slice (image: linear interpolation; labels:
    integer voxel shift) so its centroid lands on the curve. Corrects
    breath-hold misregistration between slices. Returns
    ``(image, seg, stats)``; ``stats`` holds the residual centroid error in pixels.
    """
    # Work in (slice, row, col) order, as the original SimpleITK implementation did.
    label_array = seg.array.transpose(2, 1, 0)
    image_array = image.array.transpose(2, 1, 0)

    centers = {}
    for z in range(label_array.shape[0]):
        ys, xs = np.where(label_array[z] > 0)
        centers[z] = (np.mean(xs), np.mean(ys)) if len(xs) else None
    valid = [z for z, c in centers.items() if c is not None]
    if len(valid) < 2:
        raise RealignError(f"only {len(valid)} segmented slice(s); need at least 2 to fit a centreline")

    zs = np.array(valid)
    xs = np.array([centers[z][0] for z in valid])
    ys = np.array([centers[z][1] for z in valid])
    fitted_x = np.polyval(np.polyfit(zs, xs, degree), zs)
    fitted_y = np.polyval(np.polyfit(zs, ys, degree), zs)
    fitted = {z: (fitted_x[i], fitted_y[i]) for i, z in enumerate(zs)}

    new_label = np.zeros_like(label_array)
    new_image = np.zeros_like(image_array)
    for z in range(label_array.shape[0]):
        if z not in fitted:
            new_image[z] = image_array[z]
            continue
        dx = fitted[z][0] - centers[z][0]
        dy = fitted[z][1] - centers[z][1]
        yy, xx = np.where(label_array[z] > 0)
        nx, ny = (xx + dx).astype(int), (yy + dy).astype(int)
        keep = (nx >= 0) & (nx < label_array.shape[2]) & (ny >= 0) & (ny < label_array.shape[1])
        new_label[z, ny[keep], nx[keep]] = label_array[z, yy[keep], xx[keep]]
        new_image[z] = shift(image_array[z], shift=[dy, dx], order=1, mode="constant", cval=0)

    errors = []
    for i, z in enumerate(zs):
        yy, xx = np.where(new_label[z] > 0)
        if len(xx):
            errors.append(np.hypot(np.mean(xx) - fitted_x[i], np.mean(yy) - fitted_y[i]))
    stats = {
        "mean_error_px": float(np.mean(errors)) if errors else 0.0,
        "max_error_px": float(np.max(errors)) if errors else 0.0,
        "n_slices": len(valid),
    }
    return (image.with_array(new_image.transpose(2, 1, 0)),
            seg.with_array(new_label.transpose(2, 1, 0)), stats)


def fill_apex(seg: Image, image: Image | None = None, max_slices: int = 2):
    """Close an open LV apex with up to ``max_slices`` tapered myocardium caps.

    The apex end of the stack is the end with the smaller segmented area. Empty
    slices beyond the last myocardium are filled first, then new slices are added
    (the image gets matching blank slices), each an increasingly eroded copy of the
    LV cavity labelled as myocardium. The affine is updated so added slices sit
    one slice step beyond the original stack. Returns ``(seg, image, stats)``.
    """
    lab_endo, lab_myo = CANONICAL.lv, CANONICAL.myo
    label = seg.array.transpose(2, 1, 0).astype(np.int16)
    img = image.array.transpose(2, 1, 0) if image is not None else None
    stats = {"filled": 0, "added": 0}

    n = min(3, label.shape[0] // 2)
    areas = [np.count_nonzero(label[i] > 0) for i in range(label.shape[0])]
    start = [a for a in areas[:n] if a > 0]
    end = [a for a in areas[label.shape[0] - n:] if a > 0]
    apex_at_start = (np.mean(start) if start else 0) <= (np.mean(end) if end else 0)
    stats["apex_end"] = "first" if apex_at_start else "last"

    if not apex_at_start:  # always work with the apex at index 0
        label = label[::-1].copy()
        img = img[::-1].copy() if img is not None else None

    first_myo = next((i for i in range(label.shape[0]) if (label[i] == lab_myo).any()), None)
    if (label[0] == lab_myo).any() or first_myo is None or not (label[first_myo] == lab_endo).any():
        stats["status"] = "complete" if first_myo is not None else "no myocardium"
        return seg, image, stats
    endo_mask = label[first_myo] == lab_endo

    num_filled = min(first_myo, max_slices)
    num_add = max(0, max_slices - num_filled)
    for i in range(num_filled):
        erosion = int(2 ** (1.5 * (num_filled - i)))
        label[i][binary_erosion(endo_mask, iterations=erosion)] = lab_myo
    if num_add:
        caps = []
        for i in range(num_add):
            erosion = int(2 ** (1.5 * (num_add + num_filled - i)))
            cap = np.zeros_like(label[0])
            cap[binary_erosion(endo_mask, iterations=erosion)] = lab_myo
            caps.append(cap)
        label = np.concatenate([np.array(caps), label], axis=0)
        if img is not None:
            img = np.concatenate([np.zeros((num_add,) + img.shape[1:], dtype=img.dtype), img], axis=0)

    if not apex_at_start:
        label = label[::-1].copy()
        img = img[::-1].copy() if img is not None else None

    affine = seg.affine.copy()
    if num_add and apex_at_start:  # slices were prepended: the first voxel moves back along k
        affine[:3, 3] -= num_add * affine[:3, 2]
    stats.update(filled=num_filled, added=num_add, status="filled")
    new_seg = Image(label.transpose(2, 1, 0).astype(np.uint8), affine)
    new_image = Image(img.transpose(2, 1, 0), affine.copy()) if img is not None else None
    return new_seg, new_image, stats


class Realign(Step):
    """Straighten each phase's stack along a fitted centreline (see :func:`realign`).

    Stacks with fewer than two segmented slices are left unchanged and noted in
    ``case.info["realign"]``.
    """

    name = "realign"
    writes = ("images", "segs")

    def __init__(self, degree: int = 1):
        self.degree = degree

    def process(self, case: Case) -> Case:
        notes = case.info.setdefault("realign", {})
        for phase, seg in list(case.segs.items()):
            if phase not in case.images:
                continue
            try:
                case.images[phase], case.segs[phase], notes[phase] = realign(case.images[phase], seg, self.degree)
            except RealignError as e:
                notes[phase] = f"skipped: {e}"
        return case


class FillApex(Step):
    """Close the LV apex of each phase's segmentation (see :func:`fill_apex`)."""

    name = "fill_apex"
    writes = ("images", "segs")

    def __init__(self, max_slices: int = 2):
        self.max_slices = max_slices

    def process(self, case: Case) -> Case:
        notes = case.info.setdefault("fill_apex", {})
        for phase, seg in list(case.segs.items()):
            new_seg, new_image, notes[phase] = fill_apex(seg, case.images.get(phase), self.max_slices)
            case.segs[phase] = new_seg
            if new_image is not None:
                case.images[phase] = new_image
        return case
