"""Post-processing and measurement steps on synthetic cases."""

import numpy as np
import pytest

from openheart import FillApex, Image, Measure, Realign
from openheart.measure import function_metrics, volumes
from openheart.postprocess import RealignError, fill_apex, realign

from conftest import MYO_THICKNESS_VOX, SPACING


def test_realign_puts_slice_centroids_on_a_line(case):
    seg, image = case.segs["ED"], case.images["ED"]
    shifted = seg.array.copy()
    shifted[..., 5] = np.roll(shifted[..., 5], 4, axis=0)  # one slice displaced 4 voxels
    _, new_seg, stats = realign(image, seg.with_array(shifted))
    centroids = [np.argwhere(new_seg.array[..., k] > 0).mean(0) for k in range(1, 10)]
    assert np.ptp([c[0] for c in centroids]) < 1.5
    assert stats["n_slices"] == 9


def test_realign_needs_two_slices(case):
    seg = case.segs["ED"].array.copy()
    seg[..., 2:] = 0
    with pytest.raises(RealignError):
        realign(case.images["ED"], case.segs["ED"].with_array(seg))


def test_realign_step_records_skips(case):
    case.segs["ES"] = case.segs["ES"].with_array(np.zeros_like(case.segs["ES"].array))
    case = Realign()(case)
    assert case.info["realign"]["ES"].startswith("skipped")
    assert "mean_error_px" in case.info["realign"]["ED"]


def open_apex_stack():
    """LV cavity of realistic size (radius 18 px); the apical slice has cavity but no myocardium."""
    ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    d = np.hypot(ii - 32, jj - 32)
    seg = np.zeros((64, 64, 4), np.uint8)
    seg[..., 0][d < 12] = 1
    for k in range(1, 4):
        seg[..., k][d < 21] = 2
        seg[..., k][d < 18] = 1
    affine = np.diag([1.5, 1.5, 10.0, 1.0])
    affine[:3, 3] = (5.0, 6.0, 7.0)
    return Image(seg, affine), Image(np.full(seg.shape, 100, np.int16), affine.copy())


def test_fill_apex_fills_and_prepends_caps_and_moves_origin():
    seg, image = open_apex_stack()
    new_seg, new_img, stats = fill_apex(seg, image)
    assert stats == {"filled": 1, "added": 1, "apex_end": "first", "status": "filled"}
    assert new_seg.shape[2] == seg.shape[2] + 1 == new_img.shape[2]
    assert (new_seg.array[..., 0] == 2).any() and (new_seg.array[..., 1] == 2).any()  # cap + filled slice
    assert (new_img.array[..., 0] == 0).all()  # added image slice is blank
    assert np.allclose(new_seg.affine[:3, 3], seg.affine[:3, 3] - seg.affine[:3, 2])


def test_fill_apex_at_the_last_slice_keeps_origin():
    seg, image = open_apex_stack()
    flipped = seg.with_array(seg.array[..., ::-1].copy())
    new_seg, _, stats = fill_apex(flipped, image.with_array(image.array[..., ::-1].copy()))
    assert stats["apex_end"] == "last" and stats["added"] == 1
    assert (new_seg.array[..., -1] == 2).any()
    assert np.allclose(new_seg.affine, seg.affine)


def test_fill_apex_leaves_closed_apex_alone(case):
    closed = case.segs["ED"].array.copy()
    closed[..., 0] = closed[..., 1]
    seg = case.segs["ED"].with_array(closed)
    new_seg, _, stats = fill_apex(seg)
    assert stats["status"] == "complete" and new_seg is seg


def test_volumes_and_function(case):
    voxel_ml = np.prod(SPACING) / 1000
    v = volumes(case.segs["ED"])
    assert v["lv"] == pytest.approx(np.sum(case.segs["ED"].array == 1) * voxel_ml)
    m = function_metrics(case.segs)
    assert m["lv_sv_ml"] == pytest.approx(m["lv_edv_ml"] - m["lv_esv_ml"])
    assert m["lv_ef_pct"] == pytest.approx(100 * m["lv_sv_ml"] / m["lv_edv_ml"])
    assert m["lv_mass_g"] == pytest.approx(v["myo"] * 1.05)


def test_measure_contour_thickness(case):
    pytest.importorskip("cv2")
    case = Measure(contour_thickness=True)(case)
    expected = MYO_THICKNESS_VOX * SPACING[0]
    assert case.metrics["wt2d_median_ed_mm"] == pytest.approx(expected, abs=1.5)
    assert any(k.startswith("wt2d_aha") for k in case.metrics)


def test_measure_needs_a_segmentation():
    from openheart import Case

    with pytest.raises(ValueError):
        Measure()(Case(id="empty", images={"ED": Image(np.zeros((4, 4, 4)), np.eye(4))}))


def test_fill_apex_step_updates_image_and_seg(case):
    case = FillApex()(case)
    assert set(case.info["fill_apex"]) == {"ED", "ES"}
