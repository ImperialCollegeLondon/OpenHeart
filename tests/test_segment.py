"""Segmentation with a real model — opt-in: set OPENHEART_TEST_MODEL to an nnU-Net results folder."""

import os

import numpy as np
import pytest

MODEL = os.environ.get("OPENHEART_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set OPENHEART_TEST_MODEL to run segmentation tests")


def test_segment_step_labels_every_phase(case):
    pytest.importorskip("nnunetv2")
    from openheart import Segment

    for seg in case.segs.values():
        seg.array[:] = 0
    step = Segment(MODEL, folds=os.environ.get("OPENHEART_TEST_FOLDS", "0").split(), device="cpu")
    case = step(case)
    for phase in ("ED", "ES"):
        seg = case.segs[phase]
        assert seg.shape == case.images[phase].shape
        assert np.allclose(seg.affine, case.images[phase].affine)
        assert set(np.unique(seg.array)) <= {0, 1, 2, 3}
