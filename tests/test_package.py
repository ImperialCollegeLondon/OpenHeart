"""Package surface: light import, CLI wiring, exports."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

import openheart
from openheart.cli import main
from openheart.export import qc_json
from openheart.training import export_nnunet_dataset

from conftest import make_case


def test_version():
    assert openheart.__version__


def test_core_import_is_light():
    # `import openheart` must not pull in torch, nnU-Net or VTK — they are optional extras.
    code = ("import sys, openheart, openheart.cli; "
            "sys.exit(any(m in sys.modules for m in ('torch', 'nnunetv2', 'vtk')))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 1
    assert "run" in capsys.readouterr().out


def test_cli_fit_help_reaches_subcommand(capsys):
    pytest.importorskip("torch")
    with pytest.raises(SystemExit) as exc:
        main(["fit", "--help"])
    assert exc.value.code == 0
    assert "--ed" in capsys.readouterr().out


def test_cli_custom_input_layout(tmp_path):
    from openheart import Case, LabelMap

    data = tmp_path / "data" / "s1"
    data.mkdir(parents=True)
    src = make_case()
    swapped = LabelMap(lv=3, myo=2, rv=1)
    for phase, frame in (("ED", 1), ("ES", 7)):
        src.images[phase].save(data / f"s1_frame{frame:02d}.nii.gz")
        src.segs[phase].with_array(swapped.from_canonical(src.segs[phase].array)).save(
            data / f"s1_frame{frame:02d}_gt.nii.gz")
    (data / "frames.txt").write_text("ED: 1\nES: 7\n")
    out = tmp_path / "out"
    assert main(["run", str(tmp_path / "data"), "-o", str(out), "--no-mesh", "--labels", "lv=3,myo=2,rv=1",
                 "--image", "{id}_frame{frame:02d}.nii.gz", "--seg", "{id}_frame{frame:02d}_gt.nii.gz",
                 "--frames", "frames.txt"]) == 0
    assert np.array_equal(Case.load(out / "s1").segs["ES"].array, src.segs["ES"].array)


def test_cli_run_and_qc_export(cohort_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["run", str(cohort_dir), "-o", str(out), "--layout", "openheart",
                 "--out-layout", "legacy", "--no-mesh"]) == 0
    table = list(csv.DictReader(open(out / "metrics.csv")))
    assert len(table) == 3 and all(r["status"] == "ok" for r in table)
    assert (out / "site_a/p1/seg_lvsa_SR_ED.nii.gz").exists()

    assert main(["export", "qc", str(out), "-o", str(tmp_path / "qc.json")]) == 0
    qc = json.loads((tmp_path / "qc.json").read_text())
    assert qc["__meta__"]["dataset_root"] == str(out.resolve())
    assert qc["site_b/p3"]["ED_Comments"] == "" and "lv_ef_pct" in qc["site_b/p3"]
    assert qc_json(out, tmp_path / "qc2.json") == 3


def test_export_nnunet_dataset(cohort_dir, tmp_path):
    root = export_nnunet_dataset(openheart.Cohort.load(cohort_dir), tmp_path, 7, "Phantom")
    assert root.name == "Dataset007_Phantom"
    assert len(list((root / "imagesTr").glob("*_0000.nii.gz"))) == 6  # 3 cases x ED/ES
    assert json.loads((root / "dataset.json").read_text())["numTraining"] == 6
