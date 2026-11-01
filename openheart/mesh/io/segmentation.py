"""Segmentation I/O and surface extraction.

Pure-Python replacements for the IRTK ``binarize`` + ``mcubes`` commands used by
the original 4DSegment3.0 ``CreateMeshEDES`` task. A labelled segmentation volume
(1=LV cavity, 2=myocardium, 3=RV) is turned into triangulated surfaces in world
(physical) coordinates, ready to be used as the *target* for template fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from skimage import measure

from ...image import fix_affine as _affine_with_spacing
from ...labels import LABEL_PRESETS, LabelMap

# Backwards-compatible module-level views of the default (4DSegment) convention.
LABELS = {"lv": LabelMap().lv, "myo": LabelMap().myo, "rv": LabelMap().rv}
REGION_LABELS = {
    r: LabelMap().region_labels(r) for r in ("LVendo", "LVepi", "LVmyo", "RV")
}


@dataclass
class Volume:
    """A labelled segmentation volume with its voxel->world affine."""

    data: np.ndarray  # int label map, shape (X, Y, Z)
    affine: np.ndarray  # 4x4 voxel-index -> world (mm) transform
    header: object = None
    labels: LabelMap = field(default_factory=LabelMap)  # label convention of `data`

    @classmethod
    def load(cls, path: str, labels=None) -> "Volume":
        """Load a segmentation NIfTI.

        ``labels`` sets the convention of the file (a ``LabelMap``,
        ``'lv=3,myo=2,rv=1'``, or ``None`` for the 4DSegment
        default 1=LV/2=myo/3=RV). See :meth:`LabelMap.from_spec`.
        """
        nim = nib.load(path)
        data = np.asarray(nim.dataobj)
        data = np.squeeze(data)  # drop singleton dims (some seg are X,Y,Z,1)
        return cls(
            data=np.rint(data).astype(np.int16),
            affine=_affine_with_spacing(nim),
            header=nim.header,
            labels=LabelMap.from_spec(labels),
        )

    def region_mask(self, region: str) -> np.ndarray:
        """Binary mask for a named region (union of its labels)."""
        labels = self.labels.region_labels(region)
        mask = np.isin(self.data, labels)
        return mask.astype(np.uint8)


def _voxel_to_world(verts_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map (N,3) voxel-index coordinates to world (mm) coordinates."""
    homog = np.c_[verts_vox, np.ones(len(verts_vox))]
    return (homog @ affine.T)[:, :3]


def extract_surface(
    volume: Volume,
    region: str,
    *,
    smooth_sigma: float = 2.0,
    level: float = 0.5,
    step_size: int = 1,
):
    """Extract a triangulated surface for ``region`` in world coordinates.

    Replacement for ``binarize <labels> 255 0`` + ``mcubes ... 120 -blur 2``.
    The IRTK pipeline binarises to 0/255, Gaussian-blurs (``-blur 2``), then runs
    marching cubes at iso 120. Here we mask to {0,1}, optionally smooth, and run
    marching cubes at ``level`` (0.5 == midway, the analogue of 120/255).

    Returns
    -------
    verts : (N, 3) float64 world-coordinate vertices
    faces : (M, 3) int triangle indices
    """
    binary = volume.region_mask(region).astype(np.float32)
    if binary.sum() == 0:
        raise ValueError(f"Region '{region}' is empty in this segmentation.")

    mask = binary
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        # ANISOTROPIC smoothing. ``smooth_sigma`` is in-plane voxels; scale it per
        # axis by the voxel spacing so the blur is the same PHYSICAL extent on every
        # axis. With a ~10 mm slice gap vs ~1.56 mm in-plane, an isotropic
        # sigma=2 blurs across whole z-slices and shrinks thin structures massively
        # (a [49,74,70] mm ES RV collapses to [17,17,31]); the fitted mesh then
        # faithfully follows that shrunken target and looks far too small. Scaling
        # by spacing makes z almost un-blurred (sigma ~0.3) while keeping in-plane
        # smoothing, so the surface stays both faithful in size and smooth.
        spacing = np.linalg.norm(volume.affine[:3, :3], axis=0)
        spacing = np.where(spacing > 0, spacing, spacing.max() or 1.0)
        sigma_per_axis = smooth_sigma * (spacing.min() / spacing)
        smoothed = gaussian_filter(binary, sigma=tuple(sigma_per_axis))
        # A small / thin region can still be blurred below `level` everywhere, so
        # its iso-surface vanishes and marching cubes raises "Surface level must be
        # within volume data range". Fall back to the un-smoothed 0/1 mask, which
        # always brackets level=0.5 for a non-empty region.
        mask = smoothed if smoothed.max() > level else binary

    verts_vox, faces, _normals, _values = measure.marching_cubes(
        mask, level=level, step_size=step_size, allow_degenerate=False
    )
    verts_world = _voxel_to_world(verts_vox, volume.affine)
    return verts_world.astype(np.float64), faces.astype(np.int64)
