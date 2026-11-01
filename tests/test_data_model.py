"""Labels, images, layouts, cases and cohorts."""

import nibabel as nib
import numpy as np
import pytest

from openheart import Case, Cohort, Image, LabelMap, Layout, get_layout
from openheart.image import fix_affine

from conftest import make_case, phantom_seg

SWAPPED = "lv=3,myo=2,rv=1"  # LV and RV exchanged


def test_label_presets_and_specs():
    assert LabelMap.from_spec("4dseg") == LabelMap()
    assert LabelMap.from_spec(SWAPPED).lv == 3
    assert LabelMap.from_spec((3, 2, 1)).rv == 1
    with pytest.raises(ValueError):
        LabelMap.from_spec("nonsense")


def test_label_conversion_round_trip():
    swapped = LabelMap.from_spec(SWAPPED)
    canonical = phantom_seg(9)
    on_disk = swapped.from_canonical(canonical)
    assert (on_disk[canonical == 1] == 3).all() and (on_disk[canonical == 3] == 1).all()
    assert np.array_equal(swapped.to_canonical(on_disk), canonical)


def test_image_round_trip(tmp_path, case):
    path = tmp_path / "img.nii.gz"
    case.images["ED"].save(path)
    back = Image.load(path)
    assert np.array_equal(back.array, case.images["ED"].array)
    assert back.spacing == pytest.approx((2.0, 2.0, 8.0))


def test_fix_affine_prefers_qform_over_degenerate_sform():
    nim = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.diag([2.0, 2.0, 8.0, 1.0]))
    nim.set_sform(np.eye(4), code=1)  # spacing lost, as in some clinical exports
    assert np.allclose(np.linalg.norm(fix_affine(nim)[:3, :3], axis=0), [2, 2, 8])


@pytest.mark.parametrize("layout", ["openheart", "legacy"])
def test_layout_round_trip(tmp_path, case, layout):
    case.metrics = {"lv_edv_ml": 1.5}
    case.save(tmp_path / "c", layout=layout)
    back = Case.load(tmp_path / "c", layout=layout)
    assert back.id == "c" and back.metrics == {"lv_edv_ml": 1.5}
    for phase in ("ED", "ES"):
        assert np.array_equal(back.segs[phase].array, case.segs[phase].array)
        assert np.array_equal(back.images[phase].array, case.images[phase].array)


def test_layout_label_override_is_applied_on_disk(tmp_path, case):
    case.save(tmp_path / "c", layout=get_layout("legacy", labels=SWAPPED))
    raw = Image.load(tmp_path / "c" / "seg_lvsa_SR_ED.nii.gz").array
    assert (raw[case.segs["ED"].array == 1] == 3).all()  # LV stored as 3
    back = Case.load(tmp_path / "c", layout="legacy", labels=SWAPPED)
    assert np.array_equal(back.segs["ED"].array, case.segs["ED"].array)


def test_legacy_reads_unsegmented_input(tmp_path, case):
    folder = tmp_path / "raw"
    folder.mkdir()
    case.images["ED"].save(folder / "lvsa_ED.nii.gz")
    back = Case.load(folder, layout="legacy")
    assert list(back.images) == ["ED"] and not back.segs


def test_custom_layout_with_frame_numbers(tmp_path):
    folder = tmp_path / "subject7"
    folder.mkdir()
    (folder / "frames.txt").write_text("ED: 1\nES: 9\nGroup: HCM\n")
    swapped = LabelMap.from_spec(SWAPPED)
    src = make_case()
    for phase, frame in (("ED", 1), ("ES", 9)):
        src.images[phase].save(folder / f"subject7_frame{frame:02d}.nii.gz")
        src.segs[phase].with_array(swapped.from_canonical(src.segs[phase].array)).save(
            folder / f"subject7_frame{frame:02d}_gt.nii.gz")
    layout = Layout(name="mine", image=("{id}_frame{frame:02d}.nii.gz",), seg="{id}_frame{frame:02d}_gt.nii.gz",
                    frames="frames.txt", labels=swapped)
    cohort = Cohort.load(tmp_path, layout=layout)
    assert cohort.ids == ["subject7"]
    case = cohort[0]
    assert case.info["dataset"]["Group"] == "HCM"
    assert np.array_equal(case.segs["ES"].array, src.segs["ES"].array)
    with pytest.raises(ValueError, match="read-only"):
        case.save(tmp_path / "out", layout=layout)


def test_custom_layout_patterns(tmp_path):
    folder = tmp_path / "042"
    folder.mkdir()
    src = make_case()
    for phase in ("ED", "ES"):
        src.images[phase].save(folder / f"042_SA_{phase}.nii.gz")
        src.segs[phase].save(folder / f"042_SA_{phase}_gt.nii.gz")
    layout = Layout(name="mine", image=("{id}_SA_{phase}.nii.gz",), seg="{id}_SA_{phase}_gt.nii.gz",
                    labels="4dseg", writable=False)
    cohort = Cohort.load(tmp_path, layout=layout)
    assert cohort.ids == ["042"]
    assert np.array_equal(cohort[0].segs["ED"].array, src.segs["ED"].array)


def test_custom_layout_lowercase_phase(tmp_path):
    folder = tmp_path / "001"
    folder.mkdir()
    src = make_case()
    src.images["ED"].save(folder / "001_sax_ed.nii.gz")
    src.segs["ED"].save(folder / "001_sax_ed_gt.nii.gz")
    layout = Layout(name="mine", image=("{id}_sax_{phase_lower}.nii.gz",), seg="{id}_sax_{phase_lower}_gt.nii.gz")
    cohort = Cohort.load(tmp_path, layout=layout)
    assert cohort.ids == ["001"]
    assert np.array_equal(cohort[0].segs["ED"].array, src.segs["ED"].array)


def test_custom_layout_of_images_only(tmp_path):
    folder = tmp_path / "s1"
    folder.mkdir()
    make_case().images["ED"].save(folder / "cine_ED.nii.gz")
    (tmp_path / "unrelated").mkdir()
    cohort = Cohort.load(tmp_path, layout=Layout(name="mine", image=("cine_{phase}.nii.gz",)))
    assert cohort.ids == ["s1"]
    assert list(cohort[0].images) == ["ED"] and not cohort[0].segs


def test_frame_patterns_need_a_frames_file():
    with pytest.raises(ValueError, match="frames file"):
        Layout(name="mine", image=("{id}_frame{frame:02d}.nii.gz",))


def test_cohort_ids_are_relative_paths(cohort_dir):
    cohort = Cohort.load(cohort_dir)
    assert cohort.ids == ["site_a/p1", "site_a/p2", "site_b/p3"]
    assert cohort["site_b/p3"].id == "site_b/p3"
    assert len(list(cohort)) == 3


def test_single_case_folder_is_a_cohort(cohort_dir):
    assert Cohort.load(cohort_dir / "site_a" / "p1").ids == ["p1"]


def test_save_removes_files_for_data_no_longer_held(tmp_path, case):
    case.save(tmp_path / "c")
    del case.segs["ES"]
    case.save(tmp_path / "c")
    assert not (tmp_path / "c" / "seg_ES.nii.gz").exists()
    assert (tmp_path / "c" / "seg_ED.nii.gz").exists()
