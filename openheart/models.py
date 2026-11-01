"""Trained segmentation models: resolve a model name or folder to an nnU-Net results folder."""

from __future__ import annotations

from pathlib import Path

# Published models, downloaded on first use from the Hugging Face Hub
# (pip install openheart[hub]). Each entry points at an nnU-Net training output
# folder (the one holding plans.json, dataset.json and fold_<k>/).
MODELS: dict = {}


def resolve_model(model) -> Path:
    """Return the nnU-Net results folder for a registered model name or a local path.

    A local path may be the training output folder itself
    (``.../Dataset003_RBHSHare/nnUNetTrainer__nnUNetPlans__2d``).
    """
    path = Path(model).expanduser()
    if path.is_dir():
        if not (path / "plans.json").exists():
            raise FileNotFoundError(
                f"{path} is not an nnU-Net results folder (no plans.json); point at the "
                "folder that contains plans.json, dataset.json and fold_<k>/"
            )
        return path
    if model in MODELS:
        entry = MODELS[model]
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id=entry["repo_id"], revision=entry.get("revision"))
        return Path(root) / entry.get("subfolder", "")
    known = ", ".join(MODELS) or "none published yet"
    raise FileNotFoundError(f"no model folder at {model!r} and no registered model of that name ({known})")
