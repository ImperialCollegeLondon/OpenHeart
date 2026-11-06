"""Model lookup: local results folders and published models (Hub download faked)."""

import sys
import types

import pytest

from openheart.models import MODELS, resolve_model


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        (tmp_path / "hcm").mkdir(exist_ok=True)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_local_folder(tmp_path):
    (tmp_path / "plans.json").write_text("{}")
    assert resolve_model(tmp_path) == tmp_path


def test_folder_without_plans_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="plans.json"):
        resolve_model(tmp_path)


def test_unknown_name_lists_published_models():
    with pytest.raises(FileNotFoundError, match="hcm"):
        resolve_model("no-such-model")


def test_published_model_downloads_only_requested_folds(fake_hub, tmp_path):
    assert resolve_model("hcm", folds=(1,)) == tmp_path / "hcm"
    assert fake_hub[-1]["repo_id"] == MODELS["hcm"]["repo_id"]
    assert fake_hub[-1]["allow_patterns"] == ["hcm/*.json", "hcm/fold_1/checkpoint_final.pth"]

    resolve_model("hcm", checkpoint="checkpoint_best.pth")
    assert fake_hub[-1]["allow_patterns"] == ["hcm/*.json", "hcm/fold_*/checkpoint_best.pth"]
