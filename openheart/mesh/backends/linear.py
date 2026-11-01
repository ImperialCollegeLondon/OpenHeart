"""Linear (rigid / affine) alignment — CPU, numpy/scipy.

These are faithful ports of the *initialisation* stages of the IRTK pipeline:

* ``prreg``  : rigid landmark registration (irtkPointRigidRegistration) — pure
               Kabsch: centroid-align + SVD rotation + translation, NO scale.
* ``msrreg`` : multi-surface rigid registration
               (irtkMultipleSurfaceRegistration + irtkPointRigidRegistration) —
               the same pure-rigid Kabsch fit, ICP-iterated, correspondences
               POOLED across RV + LVendo + LVepi.
* ``sareg``  : surface affine registration (irtkSurfaceRegistration +
               irtkPointAffineRegistration) — a 12-DOF affine fit GRADUALLY
               (gradient/damped, small steps), ICP-iterated. NOT a one-shot
               least-squares solve, which would crush a degenerate axis in one go.

The non-rigid B-spline stage (``snreg``) is done by the MBA backend
(``ffd_torch.py``); these linear steps give it a good starting point.

A RIGID transform has NO scale, so it cannot collapse. The affine is made
gradual (many small steps) so it eases the template onto the subject without a
one-step collapse — matching irtkPointAffineRegistration's
ConjugateGradientDescent (StepSize 0.1) rather than a one-shot lstsq.

All transforms map *template* coordinates -> *subject* coordinates (we deform
the template onto the subject), returned as 4x4 homogeneous matrices.
"""

from __future__ import annotations

import numpy as np


def _homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N,3) points."""
    homog = np.c_[points, np.ones(len(points))]
    return (homog @ T.T)[:, :3]


def rigid_from_correspondences(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best PURE-RIGID transform mapping src -> dst (Kabsch, NO scale).

    Faithful port of ``irtkPointRigidRegistration::ClosedFormOptimizer``:
    centroid-align both point sets, SVD the covariance, build the rotation, and
    if it is a reflection flip the column of V with the smallest singular value
    and recompute. Translation aligns the (rotated) centroids. There is NO scale
    — a rigid transform cannot shrink the mesh, so it cannot collapse.

    ``src`` and ``dst`` are (K,3) corresponding point sets (e.g. landmarks, or
    pooled ICP correspondences). This is the prreg / msrreg inner solver.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = src - mu_s
    D = dst - mu_d
    # Covariance H = sum target * source^T (IRTK: h += a(target) * b(source)).
    H = S.T @ D
    U, w, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) <= 0.999:
        # Reflection: flip the V column of the smallest singular value, recompute.
        i = int(np.argmin(w))
        V[:, i] *= -1
        R = V @ U.T
    t = mu_d - R @ mu_s
    return _homog(R, t)


# GPU nearest-neighbour acceleration for the ICP inner loop. Enabled when torch
# sees a CUDA device and the point sets are large enough that the GPU wins over the
# CPU KD-tree (small sets stay on the KD-tree to avoid launch overhead). Toggle off
# with HEART_MESH_GPU=0. The GPU path uses a chunked float64 cdist argmin, which
# returns the SAME nearest index as the KD-tree (no float32 tie-breaking drift).
import os as _os

_USE_GPU_NN = _os.environ.get("HEART_MESH_GPU", "1") != "0"
_GPU_NN_MIN_PTS = 2000


def _gpu_nearest(query: np.ndarray, reference: np.ndarray):
    import torch

    dev = torch.device("cuda")
    q = torch.as_tensor(query, dtype=torch.float64, device=dev)
    r = torch.as_tensor(reference, dtype=torch.float64, device=dev)
    out = torch.empty(q.shape[0], dtype=torch.int64, device=dev)
    chunk = 8192
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], r).argmin(dim=1)
    return out.cpu().numpy()


def _nearest_neighbors(query: np.ndarray, reference: np.ndarray):
    """Indices in ``reference`` nearest to each row of ``query``.

    GPU chunked-cdist when available and the sets are large; else a CPU KD-tree.
    Both return the identical nearest index (float64 throughout).
    """
    if _USE_GPU_NN and len(query) >= _GPU_NN_MIN_PTS:
        try:
            import torch

            if torch.cuda.is_available():
                return _gpu_nearest(query, reference)
        except Exception:
            pass
    from scipy.spatial import cKDTree

    tree = cKDTree(reference)
    _dist, idx = tree.query(query, k=1)
    return idx


def icp_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 0.01,
) -> np.ndarray:
    """Iterative Closest Point: PURE-RIGID fit of src points onto dst surface.

    Single-surface rigid ICP (the ``srreg`` analogue). NOT used by the default
    pipeline — that uses one joint :func:`joint_icp_rigid` per phase for all
    regions, matching 4DSegment's single msrreg-3 rigid — but kept as a public
    helper for callers needing a per-surface rigid registration.

    The single-surface msrreg analogue (irtkSurfaceRegistration::Optimize with an
    irtkPointRigidRegistration inner solver). Each iteration recomputes nearest-
    neighbour correspondences from the current transform and solves a pure-rigid
    Kabsch step. NumberOfIterations 100, Epsilon 0.01 (mirrors IRTK). Returns the
    4x4 transform mapping the original ``src`` onto ``dst``.
    """
    T = np.eye(4) if init is None else init.copy()
    moving = apply_transform(src, T)
    last_err = 0.0
    for j in range(max_iter):
        idx = _nearest_neighbors(moving, dst)
        err = float(np.mean(np.linalg.norm(moving - dst[idx], axis=1)))
        if j > 0 and last_err - err < tol:
            break
        last_err = err
        T_step = rigid_from_correspondences(moving, dst[idx])
        moving = apply_transform(moving, T_step)
        T = T_step @ T
    return T


def joint_icp_rigid(
    tpl_sets,
    tgt_sets,
    *,
    init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 0.01,
) -> np.ndarray:
    """Joint PURE-RIGID ICP over MULTIPLE region point sets (``msrreg 3``).

    Faithful port of ``irtkMultipleSurfaceRegistration`` with an
    ``irtkPointRigidRegistration`` inner solver: RV + LVendo + LVepi are
    registered *together* by one shared rigid transform. Each iteration applies
    the current ``T`` to every region's template points, finds each region's own
    nearest-neighbour correspondences in *that* region's target, then solves a
    SINGLE pure-rigid Kabsch step on the correspondences POOLED (vstacked) across
    all regions, and composes ``T = Tstep @ T``.

    There is NO scale (Kabsch only), so the joint rigid CANNOT collapse — no
    scale-freezing, no anisotropy cap, no subsampling band-aid is needed.

    ``tpl_sets`` and ``tgt_sets`` are equal-length sequences of (N_r, 3) arrays.
    Returns the 4x4 transform mapping each original template set onto its target.
    """
    tpl_sets = [np.asarray(p, float) for p in tpl_sets]
    tgt_sets = [np.asarray(p, float) for p in tgt_sets]
    if len(tpl_sets) != len(tgt_sets):
        raise ValueError("tpl_sets and tgt_sets must have the same length")
    if not tpl_sets:
        raise ValueError("joint_icp_rigid needs at least one region")

    T = np.eye(4) if init is None else init.copy()
    moving = [apply_transform(p, T) for p in tpl_sets]
    last_err = 0.0
    for j in range(max_iter):
        src_all, dst_all = [], []
        for mv, tgt in zip(moving, tgt_sets):
            idx = _nearest_neighbors(mv, tgt)
            src_all.append(mv)
            dst_all.append(tgt[idx])
        src = np.vstack(src_all)
        dst = np.vstack(dst_all)
        err = float(np.mean(np.linalg.norm(src - dst, axis=1)))
        if j > 0 and last_err - err < tol:
            break
        last_err = err
        T_step = rigid_from_correspondences(src, dst)
        moving = [apply_transform(mv, T_step) for mv in moving]
        T = T_step @ T
    return T


def affine_from_correspondences(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 12-DOF affine transform mapping src -> dst.

    The closed-form solve of one affine step. On its own (one-shot) a single
    lstsq over degenerate correspondences (e.g. a few-slice z-axis)
    flattens the template in one go, which is why :func:`icp_affine` applies only
    a damped FRACTION of this step each iteration (see there).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    X = np.c_[src, np.ones(len(src))]  # (K,4)
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)  # (4,3)
    T = np.eye(4)
    T[:3, :3] = M[:3, :].T
    T[:3, 3] = M[3, :]
    return T


def _bbox_min_ratio(moving: np.ndarray, dst: np.ndarray) -> float:
    """Smallest per-axis bbox-extent ratio of ``moving`` to ``dst``.

    A value well below 1 means an axis of ``moving`` has shrunk far smaller than
    the target's — the early warning of a degenerate-axis collapse.
    """
    mr = moving.max(0) - moving.min(0)
    dr = np.maximum(dst.max(0) - dst.min(0), 1e-6)
    return float((mr / dr).min())


def icp_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 0.01,
    step: float = 0.2,
    min_axis_floor: float = 0.7,
) -> np.ndarray:
    """ICP with a GRADUAL 12-DOF affine update each iteration (``sareg``).

    Faithful in spirit to ``irtkSurfaceRegistration::Optimize`` wrapping
    ``irtkPointAffineRegistration`` (whose inner optimiser is a gradual
    ConjugateGradientDescent with StepSize 0.1, NOT a one-shot least-squares).

    Each ICP iteration recomputes nearest-neighbour correspondences from the
    current transform, computes the closed-form lstsq affine step toward them,
    then applies only a FRACTION ``step`` of that step (interpolated from
    identity toward the full affine in matrix space). Many small steps ease the
    template onto the subject without the one-step axis collapse a full lstsq
    would cause.

    The lstsq affine of a tall template onto a few-slice (thin-z) target is
    itself degenerate, so the damped walk would, if run to its fixed point, still
    creep an axis toward zero. IRTK's bounded gradient descent halts at a local
    optimum before that; we mirror the same *don't-collapse* behaviour with a
    guard: the affine stops easing on once any axis of the moving template has
    shrunk below ``min_axis_floor`` of the target's extent (it has eased on as
    far as it safely can; the coarse FFD finishes the local fit). This replaces
    the old anisotropy cap and template-frame normalisation band-aids — those are
    NOT needed; the gradual update plus this collapse guard suffice.

    NumberOfIterations 100, Epsilon 0.01 (mirrors IRTK). Returns the 4x4
    transform mapping the original ``src`` onto ``dst``.
    """
    T = np.eye(4) if init is None else init.copy()
    moving = apply_transform(src, T)
    last_err = 0.0
    I = np.eye(4)
    for j in range(max_iter):
        idx = _nearest_neighbors(moving, dst)
        err = float(np.mean(np.linalg.norm(moving - dst[idx], axis=1)))
        if j > 0 and last_err - err < tol:
            break
        last_err = err
        T_full = affine_from_correspondences(moving, dst[idx])
        # Damped step: move only a fraction toward the full lstsq affine, in
        # matrix space (linear interpolation of the 4x4). Many small eases avoid
        # the one-shot collapse of a degenerate-axis lstsq.
        T_step = I + step * (T_full - I)
        moving_next = apply_transform(moving, T_step)
        # Collapse guard: only block a step that SHRINKS an under-floor axis
        # FURTHER. A step that GROWS the smallest axis is always safe — this is the
        # common case for a structure the rigid init under-scaled (e.g. the small
        # end-systolic LV cavity, which starts below the floor and must be allowed
        # to grow toward the target). The old guard broke immediately whenever the
        # current ratio was already below the floor, freezing the affine and
        # leaving such structures permanently too small.
        r_cur = _bbox_min_ratio(moving, dst)
        r_next = _bbox_min_ratio(moving_next, dst)
        if r_next < min_axis_floor and r_next < r_cur:
            break
        moving = moving_next
        T = T_step @ T
    return T
