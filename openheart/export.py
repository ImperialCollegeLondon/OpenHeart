"""Exports of a processed cohort for other tools."""

from __future__ import annotations

import json
from pathlib import Path

from .layouts import RECORD, get_layout


def qc_json(out_dir, path) -> int:
    """Write the review list for the segmentation QC web app (schema v2).

    ``out_dir`` must hold cases written in the ``legacy`` layout (the app reads
    ``lvsa_SR_*``, ``seg_lvsa_SR_*`` and ``vtks/F_*`` files). Keys are case folders
    relative to ``out_dir``; each entry carries the case's metrics and blank
    ``ED_Comments`` / ``ES_Comments`` review fields. Returns the number of cases.
    """
    out_dir = Path(out_dir).resolve()
    legacy = get_layout("legacy")
    entries = {}
    for folder in legacy.find(out_dir):
        record = folder / RECORD
        data = json.loads(record.read_text()) if record.exists() else {}
        key = folder.relative_to(out_dir).as_posix()
        entries[key] = {"case_id": data.get("id", folder.name), **data.get("metrics", {}),
                        "ED_Comments": "", "ES_Comments": ""}
    doc = {"__meta__": {"schema_version": 2, "dataset_root": str(out_dir)}, **entries}
    Path(path).write_text(json.dumps(doc, indent=4))
    return len(entries)
