"""Top-level cardiac mesh-fitting pipeline.

Orchestrates, per phase (ED/ES) and region (LVendo/LVepi/LVmyo/RV):

    segmentation -> target surface (marching cubes)
                 -> landmarks (subject + template)
                 -> rigid landmark init           (prreg, pure Kabsch, NO scale)
                 -> JOINT pure-rigid               (msrreg-3, RV+LVendo+LVepi)
                 -> per-region surface AFFINE       (sareg, gradual 12-DOF)
                 -> per-region B-spline FFD ds=20   (snreg, NO mesh regularisers)
    => fitted template mesh (template topology preserved)

This mirrors 4DSegment3.0's ``_sh_mesh_registration`` and the real IRTK sources:
one joint PURE-RIGID anchor (Kabsch, no scale — so it cannot collapse), then for
every region a GRADUAL 12-DOF surface affine (damped, many small steps, like
irtkPointAffineRegistration's ConjugateGradientDescent) followed by a coarse
20 mm B-spline FFD with no surface mesh penalties. There is NO scale in the
rigid stage, NO anisotropy cap, and NO template-frame normalisation — those were
self-inflicted band-aids for a scale collapse that a pure-rigid + gradual-affine
pipeline simply does not have.

The output meshes all share the bundled template's vertex count and connectivity,
so vertex *i* means the same anatomical location across every subject and phase —
which is what enables cross-case / cross-time comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import linear
from .backends.ffd_torch import FFDConfig, fit_ffd
from .errors import InsufficientLandmarksError
from .io.mesh import Mesh
from .io.segmentation import Volume, extract_surface
from .landmarks import empty_chambers, extract_landmarks
from .template import REGIONS, load_template_mesh, template_landmarks


def _gpu_enabled() -> bool:
    """True when the GPU linear-ICP path should be used (CUDA present, not opted out)."""
    import os

    if os.environ.get("HEART_MESH_GPU", "1") == "0":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _icp_affine(src, dst, **kw):
    """sareg affine ICP — GPU when available, else the numpy reference. Equivalent."""
    if _gpu_enabled():
        from .backends.ffd_gpu import icp_affine_gpu

        return icp_affine_gpu(src, dst, **kw)
    return linear.icp_affine(src, dst, **kw)


def _joint_icp_rigid(tpl_sets, tgt_sets, **kw):
    """msrreg-3 joint rigid ICP — GPU when available, else numpy. Equivalent."""
    if _gpu_enabled():
        from .backends.ffd_gpu import joint_icp_rigid_gpu

        return joint_icp_rigid_gpu(tpl_sets, tgt_sets, **kw)
    return linear.joint_icp_rigid(tpl_sets, tgt_sets, **kw)


@dataclass
class FitResult:
    """Fitted meshes + per-region quality metrics for one phase."""

    phase: str
    meshes: dict = field(default_factory=dict)  # region -> Mesh (template topology)
    metrics: dict = field(default_factory=dict)  # region -> {mean_surf_dist, ...}
    joint_rigid: object = None  # 4x4 msrreg-3 rigid actually used for this phase
    landmark_rigid: object = None  # 4x4 landmark (prreg) rigid; ES reuses ED's
    elapsed_sec: float = 0.0  # whole-subject fit time (set by fit_subject)

    def save(self, out_dir: str, prefix: str = "F"):
        import os

        os.makedirs(out_dir, exist_ok=True)
        for region, mesh in self.meshes.items():
            mesh.write(os.path.join(out_dir, f"{prefix}_{region}_{self.phase}.vtk"))


# Uniform B-spline control spacing (mm). 4DSegment runs ``snreg -ds 20`` for every
# region (mesh_ed_es.py 414-439); the snreg best-grid fix (see ffd_torch.snreg)
# makes finer spacings reliably IMPROVE the fit instead of overshooting, so error
# falls as ds shrinks. 5 mm fits tightest but starts tracking the segmentation's
# voxel staircase; 10 mm is a touch loose. 8 mm sits in between — clearly tighter
# than 10 mm (LVmyo 1.73 -> 1.63) without hugging the voxel noise the way 5 mm does.
_FFD_SPACING = 8.0

# Regions whose joint rigid anchor is built from RV + LVendo + LVepi together
# (the msrreg-3 set). LVmyo is NOT part of the anchor; it gets its own affine.
_JOINT_REGIONS = ("RV", "LVendo", "LVepi")


def compute_joint_rigid(subj_lm, tpl_lm, region_targets, landmark_init=None):
    """One joint PURE-RIGID anchor (msrreg-3) for the whole phase.

    ``region_targets`` maps region name -> (template_verts, target_verts). The
    joint set is built from whichever of RV/LVendo/LVepi are present, but REQUIRES
    both LV surfaces so the anchor is never RV-only. Returns the 4x4
    template->subject transform.

    Pure rigid throughout (Kabsch: rotation + translation, NO scale). A rigid
    transform has no scale, so it cannot shrink the mesh to a point — there is
    nothing to freeze, cap, or subsample.
    """
    # (a) Landmark init (prreg analogue): pure-rigid Kabsch on the landmarks.
    # 4DSegment initialises BOTH phases' msrreg-3 with the ED landmark rigid
    # (``-dofin landmarks_ED.dof``): the ES segmentation is small/less reliable so
    # its own landmarks drift. Pass ``landmark_init`` (ED's T0) to reuse it; the
    # surface ICP below still runs on THIS phase's own surfaces.
    T0 = landmark_init if landmark_init is not None else linear.rigid_from_correspondences(tpl_lm, subj_lm)

    present = [r for r in _JOINT_REGIONS if r in region_targets]
    if not ("LVendo" in present and "LVepi" in present):
        # Anchor needs both LV surfaces; without them fall back to the landmark
        # rigid.
        return T0

    tpl_sets = [region_targets[r][0] for r in present]
    tgt_sets = [region_targets[r][1] for r in present]
    # (b) Joint pure-rigid ICP, correspondences pooled across RV+LVendo+LVepi.
    return _joint_icp_rigid(tpl_sets, tgt_sets, init=T0)


def _align_region(region, tpl_mesh, target_verts, T_joint, ffd_config):
    """Per-region surface affine (sareg) + ds=20 FFD (snreg), template->subject.

    Mirrors the real IRTK pipeline directly in the SUBJECT frame (no template-
    frame normalisation): apply the joint rigid to the template, then fit a
    GRADUAL 12-DOF surface affine (``sareg``, damped many-small-step ICP — like
    irtkPointAffineRegistration's ConjugateGradientDescent, not a one-shot
    lstsq), then a ``snreg -ds 20`` B-spline FFD, all mapping template -> subject.
    The gradual affine eases the template onto the subject without the one-step
    axis collapse a full lstsq would cause, so NO anisotropy cap and NO
    inv(T_joint) normalisation are needed. Template topology is preserved (we only
    displace the template vertices).
    """
    # (c) Apply the joint msrreg-3 rigid anchor to the template. 4DSegment uses ONE
    # msrreg-3 rigid per phase for ALL regions (no per-region rigid refine), so we
    # apply T_joint directly and let the per-region sareg+snreg do the rest.
    pre_rigid = linear.apply_transform(tpl_mesh.verts, T_joint)

    # (d) Per-region GRADUAL surface affine (sareg) template -> subject. Many
    # small damped steps ease the template on; no collapse, no aniso cap needed.
    T_aff = _icp_affine(pre_rigid, target_verts, init=np.eye(4))
    pre = linear.apply_transform(pre_rigid, T_aff)

    # (e) snreg ds=20: deform the affine-prealigned template onto the subject
    # surface. Smoothness from the coarse 20 mm lattice + MBA only.
    fitted, info = fit_ffd(
        pre,
        tpl_mesh.faces,
        target_verts,
        ffd_config,
        control_spacing=_FFD_SPACING,
    )
    return fitted, info


def fit_phase(
    seg_path: str,
    phase: str,
    *,
    regions=REGIONS,
    labels=None,
    ffd_config: FFDConfig | None = None,
    surface_smooth_sigma: float = 2.0,
    landmark_init=None,
    verbose: bool = True,
) -> FitResult:
    """Fit all template regions to one segmentation volume (one phase).

    ``seg_path`` is a segmentation NIfTI path or an in-memory ``Volume`` (whose own
    ``labels`` then apply).

    ``labels`` sets the segmentation's label convention (a ``LabelMap``, an
    ``'lv=3,myo=2,rv=1'`` string, or ``None`` for the
    4DSegment default 1=LV/2=myo/3=RV). See ``LabelMap.from_spec``.

    ``landmark_init`` optionally supplies a precomputed landmark (prreg) rigid
    (e.g. ED's) to seed this phase's msrreg-3 instead of computing it from this
    phase's own landmarks. 4DSegment initialises the ES registration with the ED
    landmark rigid (``-dofin landmarks_ED.dof``) because the ES (end-systolic)
    segmentation is smaller / less reliable and its own landmarks drift; ES still
    runs its own surface msrreg-3. The landmark and joint rigids actually used are
    stored on the result (``.landmark_rigid`` / ``.joint_rigid``).
    """
    vol = seg_path if isinstance(seg_path, Volume) else Volume.load(seg_path, labels=labels)
    subj_lm = extract_landmarks(vol)  # uses vol.labels
    tpl_lm = template_landmarks()

    # A missing chamber yields fewer landmarks AND shifts their correspondence with
    # the template (they are appended per chamber in a fixed order), so the rigid
    # solver would either crash on the shape mismatch or align mismatched points.
    # When we are not reusing a precomputed landmark rigid (ED, or a standalone
    # phase), require the same landmark structure as the template; otherwise raise a
    # clean, catchable error so the caller can skip this subject.
    if landmark_init is None and len(subj_lm) != len(tpl_lm):
        raise InsufficientLandmarksError(
            phase, len(subj_lm), len(tpl_lm), missing=empty_chambers(vol)
        )

    result = FitResult(phase=phase)

    # Extract every requested region's target surface ONCE (avoid double marching
    # cubes), keeping the template mesh + target so the joint rigid and the
    # per-region FFD both reuse them.
    region_data = {}  # region -> (tpl_mesh, target_verts)
    for region in regions:
        tpl_mesh = load_template_mesh(region, phase)
        try:
            target_v, _ = extract_surface(vol, region, smooth_sigma=surface_smooth_sigma)
        except ValueError as e:
            if verbose:
                print(f"  [{region}] skipped: {e}")
            continue
        region_data[region] = (tpl_mesh, target_v)

    # Landmark rigid (prreg). ES reuses ED's (4DSegment: -dofin landmarks_ED.dof);
    # within ED, landmark_init is None and we compute it from this phase.
    landmark_T0 = (
        landmark_init if landmark_init is not None
        else linear.rigid_from_correspondences(tpl_lm, subj_lm)
    )
    result.landmark_rigid = landmark_T0

    # ONE msrreg-3 joint rigid for the whole phase, on THIS phase's own surfaces,
    # initialised from landmark_T0. Same single rigid is used for all 4 regions.
    joint_targets = {
        r: (region_data[r][0].verts, region_data[r][1])
        for r in _JOINT_REGIONS
        if r in region_data
    }
    T_joint = compute_joint_rigid(
        subj_lm, tpl_lm, joint_targets, landmark_init=landmark_T0
    )
    result.joint_rigid = T_joint

    for region, (tpl_mesh, target_v) in region_data.items():
        fitted, info = _align_region(
            region, tpl_mesh, target_v, T_joint, ffd_config
        )
        result.meshes[region] = Mesh(verts=fitted, faces=tpl_mesh.faces.copy())
        result.metrics[region] = info
        if verbose:
            print(
                f"  [{region}] mean={info['mean_surf_dist']:.2f}mm "
                f"p95={info['p95_surf_dist']:.2f}mm  ({tpl_mesh.n_points} pts, topology preserved)"
            )
    return result


def fit_subject(
    seg_ed: str,
    seg_es: str | None = None,
    *,
    out_dir: str | None = None,
    labels=None,
    ffd_config: FFDConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Fit a subject's ED (and optionally ES) segmentation(s).

    Parameters
    ----------
    seg_ed : ED segmentation — a NIfTI path or an in-memory ``Volume``.
    seg_es : optional ES segmentation (path or ``Volume``).
    out_dir: if given, write F_<region>_<phase>.vtk files there.
    labels : segmentation label convention (``LabelMap`` /
        ``'lv=3,myo=2,rv=1'`` / ``None`` for the 4DSegment default).

    Returns
    -------
    dict phase -> FitResult. The wall-clock fit time is printed when ``verbose``
    and is also recorded on each FitResult as ``.elapsed_sec`` (the whole-subject
    time, identical on every phase) for programmatic access.
    """
    import time

    t_start = time.perf_counter()
    results = {}
    jobs = [("ED", seg_ed)]
    if seg_es is not None:
        jobs.append(("ES", seg_es))
    ed_landmark = None
    skipped = {}  # phase -> reason (for phases that could not be fit)
    for phase, path in jobs:
        if verbose:
            print(f"Fitting {phase}" + (f" from {path}" if isinstance(path, str) else ""))
        # ES initialises its msrreg-3 with ED's LANDMARK rigid (4DSegment: ES
        # -dofin = landmarks_ED.dof), then runs its OWN surface msrreg-3 on the ES
        # surfaces. The ES segmentation is small/less reliable, so its own
        # landmarks drift; ED's landmark anchor keeps it well-initialised.
        try:
            res = fit_phase(
                path, phase, labels=labels, ffd_config=ffd_config,
                landmark_init=(ed_landmark if phase == "ES" else None), verbose=verbose,
            )
        except InsufficientLandmarksError as e:
            # Incomplete segmentation (empty chamber): skip this phase, keep going.
            if verbose:
                print(f"  SKIP {phase}: {e}")
            skipped[phase] = str(e)
            continue
        if phase == "ED":
            ed_landmark = res.landmark_rigid
        if out_dir:
            res.save(out_dir)
        results[phase] = res

    elapsed = time.perf_counter() - t_start
    for res in results.values():
        res.elapsed_sec = elapsed  # whole-subject time, recorded on each phase
    if verbose:
        if results:
            print(f"Subject fitted in {elapsed:.1f}s")
        if skipped:
            print(f"  {len(skipped)} phase(s) skipped: {', '.join(skipped)}")
    if not results:
        # Nothing could be fit (e.g. every phase had an empty chamber). Surface a
        # clear, catchable failure rather than returning silently-empty results.
        raise InsufficientLandmarksError(
            "/".join(skipped) or "subject",
            0, len(template_landmarks()),
            missing=(),
        )
    return results
