"""Synthetic short-axis cases, so the suite needs no patient data."""

import numpy as np
import pytest

from openheart import Case, Image

SPACING = (2.0, 2.0, 8.0)  # mm
SHAPE = (48, 48, 10)
MYO_THICKNESS_VOX = 3  # myocardial ring width in voxels (6 mm)


def phantom_seg(lv_radius: float, shape=SHAPE) -> np.ndarray:
    """Canonical labels: LV disc, myocardial ring, RV crescent; tapering towards k=0 (apex)."""
    seg = np.zeros(shape, dtype=np.uint8)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    centre = (shape[0] / 2, shape[1] / 2)
    for k in range(1, shape[2]):  # slice 0 stays empty, beyond the apex
        r = lv_radius * min(1.0, k / 3)
        d = np.hypot(ii - centre[0], jj - centre[1])
        rv = np.hypot(ii - centre[0], jj - (centre[1] - r - 5)) < r + 3
        seg[..., k][rv] = 3
        seg[..., k][d < r + MYO_THICKNESS_VOX] = 2
        seg[..., k][d < r] = 1
    return seg


def phantom_image(seg: np.ndarray) -> np.ndarray:
    lut = np.array([20, 200, 60, 180], dtype=np.int16)  # bg, LV blood, myo, RV blood
    return lut[seg]


def make_case(case_id="case_a", lv_ed=9.0, lv_es=6.0) -> Case:
    affine = np.diag(SPACING + (1.0,))
    case = Case(id=case_id)
    for phase, r in (("ED", lv_ed), ("ES", lv_es)):
        seg = phantom_seg(r)
        case.images[phase] = Image(phantom_image(seg), affine.copy())
        case.segs[phase] = Image(seg, affine.copy())
    return case


@pytest.fixture
def case():
    return make_case()


@pytest.fixture
def cohort_dir(tmp_path):
    """Three cases in the openheart layout under two site folders."""
    root = tmp_path / "cohort"
    for cid, (ed, es) in {"site_a/p1": (9, 6), "site_a/p2": (8, 5), "site_b/p3": (10, 7)}.items():
        make_case(cid, ed, es).save(root / cid)
    return root
