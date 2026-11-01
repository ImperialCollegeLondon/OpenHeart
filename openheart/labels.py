"""Segmentation label conventions.

Inside OpenHeart every segmentation uses one canonical convention,
``1 = LV blood pool, 2 = LV myocardium, 3 = RV blood pool`` (the 4DSegment / nnU-Net
model convention). Datasets that number their labels differently are converted once,
when a case is read or written, by its layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelMap:
    """Integer label values of the three cardiac structures in a segmentation.

    Defaults are the canonical convention (1=LV, 2=myo, 3=RV). Data with LV and RV
    swapped (1=RV, 2=myo, 3=LV), for example, is ``LabelMap(lv=3, myo=2, rv=1)``.

    The four mesh-fitting regions are derived from these (mirroring the IRTK
    ``binarize`` args in 4DSegment3.0's ``mesh_ed_es.py``):
    LVendo = {lv}, LVepi = {lv, myo}, LVmyo = {myo}, RV = {rv} (+ ``rv_extra``).
    """

    lv: int = 1  # LV blood pool / cavity
    myo: int = 2  # LV myocardium
    rv: int = 3  # RV blood pool / cavity
    rv_extra: tuple = (4,)  # extra labels merged into RV if present (some 4DSegment exports)

    def region_labels(self, region: str) -> tuple:
        """Integer labels whose union forms a named region's binary mask."""
        mapping = {
            "LVendo": (self.lv,),
            "LVepi": (self.lv, self.myo),
            "LVmyo": (self.myo,),
            "RV": (self.rv,) + tuple(self.rv_extra),
        }
        return mapping[region]

    def to_canonical(self, seg: np.ndarray) -> np.ndarray:
        """Relabel a segmentation in this convention to the canonical one."""
        out = np.zeros(seg.shape, dtype=np.uint8)
        out[seg == self.lv] = CANONICAL.lv
        out[seg == self.myo] = CANONICAL.myo
        out[np.isin(seg, (self.rv,) + tuple(self.rv_extra))] = CANONICAL.rv
        return out

    def from_canonical(self, seg: np.ndarray) -> np.ndarray:
        """Relabel a canonical segmentation into this convention."""
        out = np.zeros(seg.shape, dtype=np.uint8)
        out[seg == CANONICAL.lv] = self.lv
        out[seg == CANONICAL.myo] = self.myo
        out[seg == CANONICAL.rv] = self.rv
        return out

    @classmethod
    def from_spec(cls, spec) -> "LabelMap":
        """Build a LabelMap from a preset name, dict, sequence, or string.

        Accepts:
          * ``None`` or a ``LabelMap``     -> default / pass-through
          * a preset name                  -> ``'4dseg'``
          * a dict                         -> ``{'lv': 3, 'myo': 2, 'rv': 1}``
          * ``'lv=3,myo=2,rv=1'``          -> explicit key=value
          * ``'3,2,1'`` or ``(3, 2, 1)``   -> positional lv, myo, rv
        """
        if spec is None:
            return cls()
        if isinstance(spec, LabelMap):
            return spec
        if isinstance(spec, dict):
            return cls(**{k: int(v) for k, v in spec.items()})
        if isinstance(spec, (list, tuple)):
            lv, myo, rv = (int(x) for x in spec)
            return cls(lv=lv, myo=myo, rv=rv)
        s = str(spec).strip().lower()
        if s in LABEL_PRESETS:
            return LABEL_PRESETS[s]
        s = s.replace(" ", "")
        if "=" in s:
            try:
                kv = dict(part.split("=") for part in s.split(",") if part)
                return cls(lv=int(kv["lv"]), myo=int(kv["myo"]), rv=int(kv["rv"]))
            except (KeyError, ValueError):
                raise ValueError(
                    f"Bad label spec {spec!r}; expected 'lv=..,myo=..,rv=..'."
                ) from None
        parts = [p for p in s.split(",") if p]
        if len(parts) == 3:
            lv, myo, rv = (int(x) for x in parts)
            return cls(lv=lv, myo=myo, rv=rv)
        raise ValueError(
            f"Unrecognized label spec {spec!r}. Use a preset ({'|'.join(LABEL_PRESETS)}), "
            "'lv=3,myo=2,rv=1', or '3,2,1'."
        )


CANONICAL = LabelMap()

# Named conventions.
LABEL_PRESETS = {
    "4dseg": LabelMap(lv=1, myo=2, rv=3),  # canonical / 4DSegment3.0 / OpenHeart model
}
