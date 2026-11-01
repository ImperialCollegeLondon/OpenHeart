"""3D image volumes (MRI stacks and label maps) with their voxel-to-world geometry."""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np


def fix_affine(nim) -> np.ndarray:
    """Return a geometrically valid voxel->world affine, repairing a broken sform.

    Many clinical exports carry a DEGENERATE sform (``nim.affine`` defaults to it):
    the rotation columns are unit-length (diag ~ ±1) or even all-zero, so the voxel
    spacing is lost (e.g. z stored as 1 mm when it is really 10 mm). The true
    geometry survives in two places, tried in order:

      1. the qform matrix (``get_qform()``) — usually intact even when sform is
         broken (it keeps spacing AND direction AND origin);
      2. failing that, the header ``pixdim`` zooms — rescale each (non-zero) sform
         column to the true voxel size, keeping its direction and the origin.

    If the sform already matches the header zooms it is returned unchanged.
    """
    affine = np.asarray(nim.affine, dtype=np.float64).copy()
    try:
        zooms = np.asarray(nim.header.get_zooms()[:3], dtype=np.float64)
    except Exception:
        return affine
    if zooms.shape != (3,) or not np.all(zooms > 0):
        return affine

    col_norm = np.linalg.norm(affine[:3, :3], axis=0)
    if np.all(col_norm > 0) and np.allclose(col_norm, zooms, rtol=0.05):
        return affine  # sform already carries the right spacing

    try:
        qform = np.asarray(nim.get_qform(), dtype=np.float64)
        q_norm = np.linalg.norm(qform[:3, :3], axis=0)
        if np.all(q_norm > 0) and np.allclose(q_norm, zooms, rtol=0.05):
            return qform
    except Exception:
        pass

    if np.any(col_norm == 0):
        return affine  # no usable direction anywhere — leave as-is rather than guess
    fixed = affine.copy()
    fixed[:3, :3] = (affine[:3, :3] / col_norm) * zooms
    return fixed


@dataclass
class Image:
    """A 3D volume indexed ``array[i, j, k]`` (k = short-axis slice) plus its affine."""

    array: np.ndarray
    affine: np.ndarray  # 4x4 voxel index -> world (mm)

    @classmethod
    def load(cls, path) -> "Image":
        nim = nib.load(str(path))
        array = np.asanyarray(nim.dataobj)
        while array.ndim > 3 and array.shape[-1] == 1:  # some exports are (X, Y, Z, 1)
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError(f"{path}: expected a 3D volume, got shape {array.shape}")
        return cls(array=np.asarray(array), affine=fix_affine(nim))

    def save(self, path) -> None:
        nim = nib.Nifti1Image(self.array, self.affine)
        nim.header.set_xyzt_units("mm", "sec")
        nim.set_qform(self.affine, code=1)
        nim.set_sform(self.affine, code=1)
        nib.save(nim, str(path))

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def spacing(self) -> tuple:
        """Voxel size (mm) along i, j, k."""
        return tuple(float(s) for s in np.linalg.norm(self.affine[:3, :3], axis=0))

    def with_array(self, array: np.ndarray, affine: np.ndarray | None = None) -> "Image":
        """A new Image on the same (or a given) grid."""
        return Image(array=array, affine=self.affine.copy() if affine is None else affine)
