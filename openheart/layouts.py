"""Directory layouts: where a case's files live on disk and how its labels are numbered.

A layout reads a case folder into a :class:`~openheart.case.Case` (converting labels
to the canonical convention) and, if writable, writes a case back out. Built-in
layouts:

``openheart``  (default output)::

    <case>/image_ED.nii.gz  image_ES.nii.gz
    <case>/seg_ED.nii.gz    seg_ES.nii.gz
    <case>/meshes/<region>_<phase>.vtk      LVmyo carries the WallThickness scalar
    <case>/case.json                        metrics, step history, status

``legacy``  (the original pipeline / 4DSegment naming, read by the QC app)::

    <case>/lvsa_SR_<phase>.nii.gz   (input-only folders may hold lvsa_<phase>.nii.gz)
    <case>/seg_lvsa_SR_<phase>.nii.gz
    <case>/vtks/F_<region>_<phase>.vtk   vtks/W_LVmyo_<phase>.vtk
    <case>/mesh_features/lv_myo<phase>_wallthickness.txt

Any other dataset is read by describing its files with a :class:`Layout`::

    Layout(name="mine", image=("{id}_SA_{phase}.nii.gz",), seg="{id}_SA_{phase}_gt.nii.gz",
           labels=LabelMap(lv=3, myo=2, rv=1), writable=False)

When ED and ES are frames whose numbers differ per case, ``frames`` names a text file
in each case folder (``ED: 1``, ``ES: 12``, one per line) and the image and seg patterns
take ``{frame}``, e.g. ``"{id}_frame{frame:02d}.nii.gz"``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .image import Image
from .labels import CANONICAL, LabelMap

log = logging.getLogger(__name__)

PHASES = ("ED", "ES")
REGIONS = ("LVendo", "LVepi", "LVmyo", "RV")
THICKNESS = "WallThickness"  # point scalar carried by the LVmyo mesh after Measure
RECORD = "case.json"
CONTAINERS = ("images", "segs", "meshes", "features", "metrics")  # what a step can write


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _fields(case_name: str, phase: str, frame: int = 0) -> dict:
    """Values an image or seg filename pattern can use."""
    return {"id": case_name, "phase": phase, "phase_lower": phase.lower(), "frame": frame}


def _read_mesh(path: Path):
    from .mesh.io.mesh import Mesh  # needs the [mesh] extra

    return Mesh.read(str(path))


@dataclass(frozen=True)
class Layout:
    """File naming of one case folder. Patterns take ``{phase}``, ``{region}``, ``{id}``;
    image and seg patterns also take ``{phase_lower}``.

    With ``frames``, image and seg patterns also take ``{frame}``, read per case and
    phase from that text file; its other ``key: value`` lines go to ``case.info["dataset"]``.
    """

    name: str
    image: tuple  # candidate image filenames, first existing one wins
    seg: str = ""
    mesh: str | None = None
    thickness_mesh: str | None = None  # extra copy of LVmyo with the thickness scalar
    thickness_txt: str | None = None  # IRTK vtk2txt export of the thickness scalar
    labels: LabelMap = CANONICAL
    writable: bool = True
    frames: str | None = None  # per-case text file giving the ED and ES frame numbers

    def __post_init__(self):
        object.__setattr__(self, "labels", LabelMap.from_spec(self.labels))
        if not self.frames and any("{frame" in p for p in self._patterns):
            raise ValueError(f"layout {self.name!r}: patterns use {{frame}} but no frames file is set")

    @property
    def _patterns(self) -> tuple:
        return self.image + ((self.seg,) if self.seg else ())

    def with_labels(self, spec) -> "Layout":
        return self if spec is None else replace(self, labels=LabelMap.from_spec(spec))

    # ------------------------------------------------------------------ discover
    def is_case(self, folder: Path) -> bool:
        if self.frames:
            return (folder / self.frames).exists()
        return any((folder / p.format(**_fields(folder.name, "ED"))).exists()
                   for p in self._patterns)

    def find(self, root) -> list:
        """Case folders under ``root`` (``root`` itself counts if it is a case)."""
        root = Path(root)
        if self.frames:
            return sorted(p.parent for p in root.rglob(self.frames))
        found = {root} if self.is_case(root) else set()
        for pattern in self._patterns:
            glob = pattern.format(**_fields("*", "ED"))
            found.update(p.parent for p in root.rglob(glob))
        return sorted(found)

    # ---------------------------------------------------------------------- read
    def _read_frames(self, folder: Path) -> dict:
        if not self.frames:
            return {}
        return dict(re.findall(r"^(\w+):\s*(\S+)", (folder / self.frames).read_text(), re.M))

    def _image_path(self, folder: Path, fields: dict):
        for pattern in self.image:
            path = folder / pattern.format(**fields)
            if path.exists():
                return path
        return None

    def read(self, folder, case_id: str | None = None):
        from .case import Case

        folder = Path(folder)
        case = Case(id=case_id or folder.name, source=str(folder))
        frames = self._read_frames(folder)
        for phase in PHASES:
            if self.frames and phase not in frames:
                continue
            fields = _fields(folder.name, phase, int(frames.get(phase, 0)))
            image_path = self._image_path(folder, fields)
            if image_path is not None:
                case.images[phase] = Image.load(image_path)
            seg_path = folder / self.seg.format(**fields) if self.seg else None
            if seg_path is not None and seg_path.exists():
                seg = Image.load(seg_path)
                case.segs[phase] = seg.with_array(self.labels.to_canonical(np.rint(seg.array)))
            if self.mesh:
                paths = {r: folder / self.mesh.format(region=r, phase=phase) for r in REGIONS}
                if any(p.exists() for p in paths.values()):
                    case.meshes[phase] = {r: _read_mesh(p) for r, p in paths.items() if p.exists()}
                thick = folder / self.thickness_mesh.format(phase=phase) if self.thickness_mesh else None
                if thick is not None and thick.exists():
                    case.features.setdefault(phase, {})["LVmyo"] = dict(_read_mesh(thick).point_scalars or {})
                for region, mesh in case.meshes.get(phase, {}).items():
                    if mesh.point_scalars:  # per-vertex measurements live in case.features
                        case.features.setdefault(phase, {}).setdefault(region, {}).update(mesh.point_scalars)
                        mesh.point_scalars = None
        record = folder / RECORD
        data = json.loads(record.read_text()) if record.exists() else {}
        case.metrics = data.get("metrics", {})
        case.info = data.get("info", {})
        dataset = {k: v for k, v in frames.items() if k not in PHASES}
        if dataset:
            case.info["dataset"] = dataset
        case.history = data.get("history", [])
        case.origin = data.get("origin") or [c for c in CONTAINERS if getattr(case, c)]
        if not case.images and not case.segs:
            raise FileNotFoundError(f"{folder}: no {self.name}-layout images or segmentations found")
        return case

    # --------------------------------------------------------------------- write
    def write(self, case, folder, **record) -> Path:
        """Write everything the case holds into ``folder``; ``record`` adds case.json fields."""
        if not self.writable or self.frames:
            raise ValueError(f"layout {self.name!r} is read-only; write with 'openheart' or 'legacy'")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._remove_stale(case, folder)
        for phase, image in case.images.items():
            image.save(folder / self.image[0].format(**_fields(folder.name, phase)))
        for phase, seg in case.segs.items():
            out = seg.with_array(self.labels.from_canonical(seg.array))
            out.save(folder / self.seg.format(**_fields(folder.name, phase)))
        for phase, meshes in case.meshes.items():
            for region, mesh in meshes.items():
                scalars = case.features.get(phase, {}).get(region) or None
                with_scalars = type(mesh)(verts=mesh.verts, faces=mesh.faces, point_scalars=scalars)
                if self.thickness_mesh and scalars and THICKNESS in scalars:
                    path = folder / self.thickness_mesh.format(phase=phase)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with_scalars.write(str(path))
                    if self.thickness_txt:
                        txt = folder / self.thickness_txt.format(phase=phase)
                        txt.parent.mkdir(parents=True, exist_ok=True)
                        with_scalars.write_txt(str(txt), scalar=THICKNESS)
                    with_scalars = mesh  # F_ meshes carry no scalars, as in 4DSegment
                path = folder / self.mesh.format(region=region, phase=phase)
                path.parent.mkdir(parents=True, exist_ok=True)
                with_scalars.write(str(path))
        data = {"id": case.id, "source": case.source, "origin": case.origin, "history": case.history,
                "metrics": case.metrics, "info": case.info, **record}
        (folder / RECORD).write_text(json.dumps(data, indent=2, default=_json_default))
        return folder


    def _remove_stale(self, case, folder: Path) -> None:
        """Delete this layout's files for data the case no longer holds (e.g. undone steps)."""
        stale = []
        for phase in PHASES:
            if phase not in case.images:
                stale += [folder / p.format(**_fields(folder.name, phase)) for p in self.image]
            if phase not in case.segs:
                stale.append(folder / self.seg.format(**_fields(folder.name, phase)))
            meshes = case.meshes.get(phase, {})
            if self.mesh:
                stale += [folder / self.mesh.format(region=r, phase=phase) for r in REGIONS if r not in meshes]
            if THICKNESS not in case.features.get(phase, {}).get("LVmyo", {}):
                stale += [folder / p.format(phase=phase)
                          for p in (self.thickness_mesh, self.thickness_txt) if p]
        for path in stale:
            if path.is_file():
                path.unlink()


LAYOUTS = {
    "openheart": Layout(
        name="openheart",
        image=("image_{phase}.nii.gz",),
        seg="seg_{phase}.nii.gz",
        mesh="meshes/{region}_{phase}.vtk",
    ),
    "legacy": Layout(
        name="legacy",
        image=("lvsa_SR_{phase}.nii.gz", "lvsa_{phase}.nii.gz"),
        seg="seg_lvsa_SR_{phase}.nii.gz",
        mesh="vtks/F_{region}_{phase}.vtk",
        thickness_mesh="vtks/W_LVmyo_{phase}.vtk",
        thickness_txt="mesh_features/lv_myo{phase}_wallthickness.txt",
    ),
}


def get_layout(layout="openheart", labels=None) -> Layout:
    """Resolve a layout name (or pass a Layout through), optionally overriding its labels."""
    if isinstance(layout, Layout):
        return layout.with_labels(labels)
    try:
        return LAYOUTS[layout].with_labels(labels)
    except KeyError:
        raise ValueError(f"unknown layout {layout!r}; choose from {', '.join(LAYOUTS)}") from None
