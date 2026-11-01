"""Smoke tests for the fitting pipeline. Run with: pytest -q

Uses the bundled template's own ED segmentation as a (trivial) subject so the test
needs no external data — it verifies the pipeline runs and, crucially, that output
topology equals the template topology.
"""

import shutil

import numpy as np
import pytest

pytest.importorskip("torch")

import openheart.mesh as hmf
from openheart.mesh import features
from openheart.mesh.template import (
    TEMPLATE_SEG_ED,
    load_template_mesh,
    template_landmarks,
)
from openheart.mesh.io.segmentation import Volume, extract_surface


def test_template_assets_present():
    for region in hmf.REGIONS:
        for phase in hmf.PHASES:
            m = load_template_mesh(region, phase)
            assert m.n_points > 0 and m.n_faces > 0


def test_landmarks_shape():
    lm = template_landmarks()
    assert lm.shape == (7, 3)


def test_surface_extraction():
    vol = Volume.load(TEMPLATE_SEG_ED)
    v, f = extract_surface(vol, "LVepi")
    assert v.shape[1] == 3 and f.shape[1] == 3 and len(v) > 100


def test_fit_preserves_topology():
    cfg = hmf.FFDConfig(iterations=20, control_spacing=20.0, device="cpu")
    res = hmf.fit_phase(TEMPLATE_SEG_ED, "ED", regions=("LVepi",), ffd_config=cfg, verbose=False)
    assert "LVepi" in res.meshes
    fitted = res.meshes["LVepi"]
    tpl = load_template_mesh("LVepi", "ED")
    assert fitted.n_points == tpl.n_points
    assert fitted.n_faces == tpl.n_faces
    assert np.array_equal(fitted.faces, tpl.faces)


def test_wall_thickness_length():
    endo = load_template_mesh("LVendo", "ED")
    epi = load_template_mesh("LVepi", "ED")
    wt = features.wall_thickness(endo, epi)
    assert len(wt) == endo.n_points
    assert np.all(wt >= 0)


def test_wall_thickness_myocardium_mesh(tmp_path):
    # Faithful 4Dsegment: thickness scalar carried on the LVmyo mesh
    # (endo/epi/myo have different vertex counts — mapping is by nearest point).
    endo = load_template_mesh("LVendo", "ED")
    epi = load_template_mesh("LVepi", "ED")
    myo = load_template_mesh("LVmyo", "ED")
    out = features.wall_thickness_myocardium(endo, epi, myo)
    # Geometry is the myo mesh; scalar named WallThickness, one per myo vertex.
    assert out.n_points == myo.n_points
    assert np.array_equal(out.faces, myo.faces)
    wt = out.point_scalars["WallThickness"]
    assert wt.shape == (myo.n_points,)
    assert np.all(wt >= 0)
    # Scalar survives the VTK write/read round-trip (downstream tools read it back).
    from openheart.mesh.io.mesh import Mesh

    p = tmp_path / "W_LVmyo_ED.vtk"
    out.write(str(p))
    back = Mesh.read(str(p))
    assert "WallThickness" in back.point_scalars
    assert np.allclose(back.point_scalars["WallThickness"], wt, atol=1e-6)


def _wall_thickness_myo_reference(endo, epi, myo):
    """Brute-force transcription of cardiacwallthickness.cc (-myocardium)."""
    E, P, M = endo.verts, epi.verts, myo.verts
    array_target = np.array([np.sqrt(((P - E[i]) ** 2).sum(1)).min() for i in range(len(E))])
    array_source = np.array([np.sqrt(((E - P[i]) ** 2).sum(1)).min() for i in range(len(P))])
    out = np.empty(len(M))
    for i in range(len(M)):
        de = np.sqrt(((E - M[i]) ** 2).sum(1)); ie = de.argmin()  # myo -> endo
        dp = np.sqrt(((P - M[i]) ** 2).sum(1)); ip = dp.argmin()  # myo -> epi
        out[i] = array_source[ip] if dp[ip] < de[ie] else array_target[ie]
    return out


def test_wall_thickness_myocardium_matches_irtk_cc():
    # Bit-for-bit agreement with the original IRTK cardiacwallthickness.cc, except
    # at exact-distance ties (equidistant neighbours), where cKDTree vs VTK's
    # locator may pick differently — a known, vanishingly rare tolerance.
    endo = load_template_mesh("LVendo", "ES")  # small enough for brute force
    epi = load_template_mesh("LVepi", "ES")
    myo = load_template_mesh("LVmyo", "ES")
    ref = _wall_thickness_myo_reference(endo, epi, myo)
    mine = features.wall_thickness_myocardium(endo, epi, myo).point_scalars["WallThickness"]
    differing = np.flatnonzero(np.abs(ref - mine) > 1e-6)
    # Every differing vertex must be an exact tie: its two nearest endo (or epi)
    # vertices are equidistant. Otherwise it's a real algorithm bug.
    for i in differing:
        de = np.sort(np.sqrt(((endo.verts - myo.verts[i]) ** 2).sum(1)))
        dp = np.sort(np.sqrt(((epi.verts - myo.verts[i]) ** 2).sum(1)))
        assert (de[1] - de[0] < 1e-9) or (dp[1] - dp[0] < 1e-9), f"vertex {i} differs but is not a tie"
    assert len(differing) <= 5  # observed: 3/13646 for ES, 0 for ED


# --- configurable label conventions -----------------------------------------


def _relabel(path, tmpdir, *, lv, myo, rv):
    """Rewrite the template seg (4DSeg convention) with arbitrary label integers.

    Identifies LV/myo/RV structures via the *default* convention (the template's
    own RV is actually label 4), then writes them out under the requested ints, so
    a Volume loaded with a matching LabelMap should be byte-for-byte equivalent.
    """
    import os

    import nibabel as nib

    from openheart.mesh.io.segmentation import LabelMap

    dflt = LabelMap()
    nim = nib.load(path)
    d = np.rint(np.asarray(nim.dataobj)).astype(np.int16)
    out = np.zeros_like(d)
    out[np.isin(d, (dflt.lv,))] = lv
    out[np.isin(d, (dflt.myo,))] = myo
    out[np.isin(d, (dflt.rv,) + tuple(dflt.rv_extra))] = rv
    p = os.path.join(tmpdir, "relabel.nii.gz")
    nib.save(nib.Nifti1Image(out.astype(np.uint8), nim.affine), p)
    return p


def _drop_rv(path, tmpdir):
    """Copy the template seg with the RV structure removed (mapped to background).

    The template's RV is label 4 (see _relabel), so an incomplete segmentation with
    an empty RV is simulated by zeroing LV/myo-preserving everything else.
    """
    import os

    import nibabel as nib

    from openheart.mesh.io.segmentation import LabelMap

    dflt = LabelMap()
    nim = nib.load(path)
    d = np.rint(np.asarray(nim.dataobj)).astype(np.int16)
    d[np.isin(d, (dflt.rv,) + tuple(dflt.rv_extra))] = 0  # remove RV
    p = os.path.join(tmpdir, "seg_noRV_ED.nii.gz")
    nib.save(nib.Nifti1Image(d.astype(np.uint8), nim.affine), p)
    return p


def test_insufficient_landmarks_raises_cleanly(tmp_path):
    # An incomplete segmentation (empty RV) must raise a clean, catchable error
    # instead of crashing in the rigid solver (the old ValueError from S.T @ D).
    import openheart.mesh as hmf
    from openheart.mesh import InsufficientLandmarksError
    from openheart.mesh.landmarks import empty_chambers, extract_landmarks
    from openheart.mesh.io.segmentation import Volume

    seg = _drop_rv(TEMPLATE_SEG_ED, str(tmp_path))
    vol = Volume.load(seg)
    # Fewer landmarks than the template, and RV flagged as the empty chamber.
    assert len(extract_landmarks(vol)) < 7
    assert "RV" in empty_chambers(vol)

    cfg = hmf.FFDConfig(iterations=5, control_spacing=20.0, device="cpu")
    with pytest.raises(InsufficientLandmarksError):
        hmf.fit_phase(seg, "ED", ffd_config=cfg, verbose=False)
    # fit_subject (ED-only here) surfaces the same catchable error, not a traceback.
    with pytest.raises(InsufficientLandmarksError):
        hmf.fit_subject(seg, ffd_config=cfg, verbose=False)


def test_labelmap_from_spec():
    from openheart.mesh import LabelMap

    assert LabelMap.from_spec(None) == LabelMap()
    assert LabelMap.from_spec("4dseg") == LabelMap()
    assert LabelMap.from_spec("lv=3,myo=2,rv=1").lv == 3
    assert LabelMap.from_spec({"lv": 3, "myo": 2, "rv": 1}) == LabelMap(lv=3, myo=2, rv=1)
    assert LabelMap.from_spec((3, 2, 1)) == LabelMap(lv=3, myo=2, rv=1)
    with pytest.raises(ValueError):
        LabelMap.from_spec("nonsense")


def test_configurable_labels_region_mask_equivalence(tmp_path):
    # Region masks must be identical whether we read the template under its native
    # convention or read a relabelled copy under a matching custom LabelMap — i.e.
    # configuring labels removes any need to relabel the data externally.
    vol_default = Volume.load(TEMPLATE_SEG_ED)
    custom = _relabel(TEMPLATE_SEG_ED, str(tmp_path), lv=7, myo=5, rv=9)
    vol_custom = Volume.load(custom, labels="lv=7,myo=5,rv=9")
    for region in hmf.REGIONS:
        assert np.array_equal(
            vol_default.region_mask(region), vol_custom.region_mask(region)
        ), region


def test_swapped_labels_equivalence(tmp_path):
    # LV/RV-swapped data read with a matching spec == default on 4DSeg data.
    vol_default = Volume.load(TEMPLATE_SEG_ED)
    swapped = _relabel(TEMPLATE_SEG_ED, str(tmp_path), lv=3, myo=2, rv=1)
    vol_swapped = Volume.load(swapped, labels="lv=3,myo=2,rv=1")
    for region in hmf.REGIONS:
        assert np.array_equal(
            vol_default.region_mask(region), vol_swapped.region_mask(region)
        ), region


def test_configurable_labels_landmarks_equivalence(tmp_path):
    from openheart.mesh.landmarks import extract_landmarks

    lm_default = extract_landmarks(Volume.load(TEMPLATE_SEG_ED))
    swapped = _relabel(TEMPLATE_SEG_ED, str(tmp_path), lv=3, myo=2, rv=1)
    lm_swapped = extract_landmarks(Volume.load(swapped, labels="lv=3,myo=2,rv=1"))
    assert lm_default.shape == lm_swapped.shape == (7, 3)
    assert np.allclose(lm_default, lm_swapped)


def _coarse_template(folder):
    """A template with its own topology: the bundled one with half of its faces."""
    from openheart.mesh.io.mesh import Mesh

    (folder / "vtks").mkdir(parents=True)
    shutil.copy(TEMPLATE_SEG_ED, folder / "segmentation_ED.nii.gz")
    for region in hmf.REGIONS:
        for phase in ("ED", "ES"):
            mesh = load_template_mesh(region, phase)
            used, faces = np.unique(mesh.faces[: len(mesh.faces) // 2], return_inverse=True)
            coarse = Mesh(verts=mesh.verts[used], faces=faces.reshape(-1, 3))
            coarse.write(str(folder / "vtks" / f"{region}_{phase}.vtk"))
    return folder


def test_fitmesh_with_custom_template(tmp_path):
    from openheart import Case, FitMesh, Image

    template = _coarse_template(tmp_path / "template")
    seg = Image.load(TEMPLATE_SEG_ED)
    case = FitMesh(iterations=5, device="cpu", template=template)(Case(id="t", segs={"ED": seg}))
    for region in hmf.REGIONS:
        expected = load_template_mesh(region, "ED", str(template))
        assert case.meshes["ED"][region].n_points == expected.n_points < load_template_mesh(region, "ED").n_points
    assert FitMesh(template=template).signature != FitMesh().signature


def test_template_folder_is_checked(tmp_path):
    from openheart import FitMesh

    with pytest.raises(FileNotFoundError, match="template"):
        FitMesh(template=tmp_path)
    with pytest.raises(FileNotFoundError, match="template mesh not found"):
        load_template_mesh("RV", "ED", str(tmp_path))


def test_fitmesh_and_measure_steps(tmp_path):
    from openheart import Case, FitMesh, Image, Measure, Pipeline

    seg = Image.load(TEMPLATE_SEG_ED)
    case = Case(id="template", segs={"ED": seg})
    case = Pipeline([FitMesh(iterations=5, device="cpu"), Measure()])(case)
    assert set(case.meshes["ED"]) == set(hmf.REGIONS)
    assert case.meshes["ED"]["LVepi"].n_points == load_template_mesh("LVepi", "ED").n_points
    wt = case.features["ED"]["LVmyo"]["WallThickness"]
    assert len(wt) == case.meshes["ED"]["LVmyo"].n_points
    assert 0 < case.metrics["wt_mean_ed_mm"] < case.metrics["wt_max_ed_mm"]

    for layout in ("openheart", "legacy"):  # meshes and the thickness field survive a round trip
        case.save(tmp_path / layout, layout=layout)
        back = Case.load(tmp_path / layout, layout=layout)
        assert np.allclose(back.features["ED"]["LVmyo"]["WallThickness"], wt)
        assert back.meshes["ED"]["LVmyo"].point_scalars is None
    assert (tmp_path / "legacy" / "vtks" / "W_LVmyo_ED.vtk").exists()
    assert (tmp_path / "legacy" / "mesh_features" / "lv_myoED_wallthickness.txt").exists()
