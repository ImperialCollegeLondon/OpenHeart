"""The data model: one :class:`Case` per subject, a :class:`Cohort` of many."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .layouts import PHASES, get_layout

log = logging.getLogger(__name__)


@dataclass
class Case:
    """Everything known about one subject, passed from step to step.

    ``images`` / ``segs`` map a phase (``"ED"``, ``"ES"``) to an
    :class:`~openheart.image.Image`; segmentations always use the canonical labels
    (1=LV, 2=myo, 3=RV). ``meshes`` maps a phase to ``{region: Mesh}`` and
    ``features`` to ``{region: {name: per-vertex array}}`` (e.g. LVmyo
    WallThickness). ``metrics`` is a flat ``{name: number}`` dict (one row of the
    cohort table) and ``info`` holds per-step diagnostics.

    ``history`` lists the steps applied so far (signature, what each wrote, the
    metrics it added) and ``origin`` what the source folder provided; together
    they let an interrupted or re-configured run resume exactly.
    """

    id: str
    images: dict = field(default_factory=dict)
    segs: dict = field(default_factory=dict)
    meshes: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    origin: list = field(default_factory=list)
    source: str | None = None

    @classmethod
    def load(cls, folder, layout="openheart", labels=None, case_id: str | None = None) -> "Case":
        """Read one case folder written in ``layout`` (see :mod:`openheart.layouts`)."""
        return get_layout(layout, labels).read(folder, case_id=case_id)

    def save(self, folder, layout="openheart", **record) -> Path:
        """Write the case into ``folder``; extra keyword fields go into ``case.json``."""
        return get_layout(layout).write(self, folder, **record)

    @property
    def phases(self) -> tuple:
        return tuple(p for p in PHASES if p in self.images or p in self.segs)

    def __repr__(self) -> str:
        parts = [f"images={list(self.images)}", f"segs={list(self.segs)}"]
        if self.meshes:
            parts.append(f"meshes={ {p: list(m) for p, m in self.meshes.items()} }")
        if self.metrics:
            parts.append(f"metrics={len(self.metrics)}")
        return f"Case({self.id!r}, {', '.join(parts)})"


class Cohort:
    """The case folders under a root directory, loaded lazily one case at a time.

    Case ids are folder paths relative to the root (``center_A/000123_1``), so
    nested site/batch folders stay distinct and are mirrored in the output.
    """

    def __init__(self, root, folders, layout="openheart", labels=None):
        self.root = Path(root)
        self.layout = get_layout(layout, labels)
        self.folders = list(folders)

    @classmethod
    def load(cls, root, layout="openheart", labels=None) -> "Cohort":
        lay = get_layout(layout, labels)
        folders = lay.find(root)
        log.info("found %d %s-layout cases under %s", len(folders), lay.name, root)
        return cls(root, folders, lay)

    def case_id(self, folder: Path) -> str:
        rel = Path(folder).resolve().relative_to(self.root.resolve())
        return rel.as_posix() if rel.parts else Path(folder).name

    @property
    def ids(self) -> list:
        return [self.case_id(f) for f in self.folders]

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, key) -> Case:
        if isinstance(key, str):
            key = self.ids.index(key)
        folder = self.folders[key]
        return self.layout.read(folder, case_id=self.case_id(folder))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"Cohort({str(self.root)!r}, {len(self)} cases, layout={self.layout.name!r})"
