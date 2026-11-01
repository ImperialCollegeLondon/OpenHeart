"""Exceptions raised by the fitting pipeline."""

from __future__ import annotations


class FitError(Exception):
    """Base class for recoverable, per-subject/per-phase fitting failures."""


class InsufficientLandmarksError(FitError):
    """A segmentation lacks the landmarks needed to initialise the rigid alignment.

    Raised when a phase's segmentation does not yield the same landmark structure
    as the template — e.g. a chamber (LV / myo / RV) is empty, so its base/mid/apex
    landmarks are missing. Landmarks are appended per chamber in a fixed order, so a
    missing chamber does not just reduce the count, it *shifts the correspondence*
    with the template — the subject cannot be reliably registered and must be
    skipped rather than fit against mismatched points.
    """

    def __init__(self, phase: str, n_subject: int, n_template: int, missing=()):
        self.phase = phase
        self.n_subject = n_subject
        self.n_template = n_template
        self.missing = tuple(missing)
        miss = f" (missing/empty: {', '.join(self.missing)})" if self.missing else ""
        super().__init__(
            f"{phase}: segmentation yields {n_subject} landmarks but the template "
            f"has {n_template}{miss}; cannot initialise rigid alignment — skipping."
        )
