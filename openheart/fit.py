"""Template mesh fitting as a pipeline step (wraps :mod:`openheart.mesh`)."""

from __future__ import annotations

from .case import Case
from .pipeline import Step


class FitMesh(Step):
    """Fit the cardiac template to the ED (and ES) segmentation.

    Adds ``case.meshes[phase] = {"LVendo", "LVepi", "LVmyo", "RV": Mesh}``; every
    mesh has the template's topology, so vertex *i* is the same anatomy in every
    case. Per-region surface distances go to ``case.info["fit"]``. ES is initialised
    from ED's landmark registration, as in 4DSegment3.0.
    """

    name = "fit_mesh"
    volatile = ("device",)
    writes = ("meshes",)

    def __init__(self, iterations: int = 100, control_spacing: float = 20.0, device: str = "cuda"):
        self.iterations = iterations
        self.control_spacing = control_spacing
        self.device = device

    def setup(self):
        from .mesh.backends.ffd_torch import FFDConfig

        self._config = FFDConfig(iterations=self.iterations, control_spacing=self.control_spacing,
                                 device=self.device)

    def process(self, case: Case) -> Case:
        from .labels import CANONICAL
        from .mesh.fitting import fit_subject
        from .mesh.io.segmentation import Volume

        if "ED" not in case.segs:
            raise ValueError(f"case {case.id} has no ED segmentation to fit")
        volumes = {p: Volume(data=s.array.astype("int16"), affine=s.affine, labels=CANONICAL)
                   for p, s in case.segs.items()}
        results = fit_subject(volumes["ED"], volumes.get("ES"), ffd_config=self._config, verbose=False)
        for phase, res in results.items():
            case.meshes[phase] = dict(res.meshes)
        case.info["fit"] = {
            phase: {region: {k: round(float(v), 3) for k, v in m.items() if isinstance(v, (int, float))}
                    for region, m in res.metrics.items()}
            for phase, res in results.items()
        }
        skipped = [p for p in volumes if p not in results]
        if skipped:
            case.info["fit"]["skipped"] = skipped
        return case
