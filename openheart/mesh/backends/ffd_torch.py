"""Non-rigid template fitting via cubic B-spline Free-Form Deformation (IRTK snreg).

This is a faithful numpy port of the IRTK ``snreg`` surface-registration pipeline:

    irtkSurfaceRegistration::Optimize        (outer ICP, irtkSurfaceRegistration.cc:115)
      -> irtkPointFreeFormRegistration::Optimize   (residual fit, .cc:71)
        -> irtkBSplineFreeFormTransformation3D::Approximate3D  (bspline3d.cc:1926)

There is NO gradient descent, NO chamfer loss, and NO mesh Laplacian/edge/normal
penalty.  The deformation model is a cubic-B-spline FFD over a coarse (20 mm)
control lattice built across the TARGET (template) bounding box, padded by one
spacing on each side.  Smoothness comes entirely from the multilevel-B-spline
approximation (MBA): each ICP iteration splats the residual onto the control
lattice weighted by ``B_I^2`` and normalised by ``Sum B_I^2`` (Lee et al.),
incrementally refining the field.  Template topology / point correspondence is
preserved because we only displace the fixed template vertices through the smooth
FFD; we never re-mesh.

The control lattice carries the (_z,_y,_x) logical control points but is allocated
over a PADDED range [-2 .. _x+1] etc. (4 extra planes per axis); a logical index I
maps to array index I+2.  Everything is float64 (IRTK is double).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FFDConfig:
    """snreg parameters (replaces the old Adam/chamfer config)."""

    control_spacing: float = 20.0  # mm between control points (snreg -ds 20)
    iterations: int = 100          # outer ICP iterations
    epsilon: float = 0.01          # mm; stop when error improves by less than this
    symmetric: bool = True         # snreg -symmetric (4DSegment uses it)
    verbose: bool = False
    # "cuda" (use GPU when available — the default), "cpu", or "auto". The GPU and
    # CPU snreg paths are numerically identical; set HEART_MESH_GPU=0 to force CPU.
    device: str = "cuda"


# --------------------------------------------------------------------------- #
# Cubic B-spline basis (uniform).  For fractional u in [0,1):
#   B0(u) = (1-u)^3 / 6
#   B1(u) = (3u^3 - 6u^2 + 4)/6
#   B2(u) = (-3u^3 + 3u^2 + 3u + 1)/6
#   B3(u) = u^3 / 6
# Matches IRTK's LookupTable[.] (analytic, no LUT needed -> more accurate).
# --------------------------------------------------------------------------- #
def _bspline_weights(u: np.ndarray) -> np.ndarray:
    """Uniform cubic B-spline basis for fractional offsets ``u`` in [0,1).

    Returns (...,4) weights B0..B3 for the 4 contributing control points.
    """
    u = np.asarray(u, dtype=np.float64)
    u2 = u * u
    u3 = u2 * u
    w0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0
    w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
    w2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
    w3 = u3 / 6.0
    return np.stack([w0, w1, w2, w3], axis=-1)


class BSplineFFD:
    """Cubic B-spline FFD over a 3D control lattice (numpy / float64).

    Built exactly like the IRTK ctor in snreg.cc + bspline3d.cc:386-455:
    bbox expanded by one spacing each side, dims = round(span/ds)+1, spacing
    re-snapped so the box endpoints are control-point centres.
    """

    def __init__(self, lo, hi, spacing=20.0):
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        ds = float(spacing)

        # snreg.cc:249-255 — expand the target bbox by one spacing on each side.
        self.start_box = lo - ds            # lattice coord (0,0,0) sits here
        self.end_box = hi + ds

        span = self.end_box - self.start_box
        # bspline3d.cc:418-421 — control-point counts.
        self._x = int(round(span[0] / ds)) + 1
        self._y = int(round(span[1] / ds)) + 1
        self._z = int(round(span[2] / ds)) + 1
        self._x = max(self._x, 2)
        self._y = max(self._y, 2)
        self._z = max(self._z, 2)

        # bspline3d.cc:449 — re-snap spacing so endpoints are CP centres.
        self._d = np.array([
            span[0] / (self._x - 1),
            span[1] / (self._y - 1),
            span[2] / (self._z - 1),
        ], dtype=np.float64)

        # Control storage, padded by 2 planes each side.  Logical CP (I,J,K) with
        # I in [-2 .. _x+1] -> self._data[K+2, J+2, I+2].  Zero-init = identity.
        self._data = np.zeros((self._z + 4, self._y + 4, self._x + 4, 3),
                              dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Coordinate transforms.  Identity axes => lattice = (world-start)/d.
    # ------------------------------------------------------------------ #
    def WorldToLattice(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - self.start_box) / self._d

    def LatticeToWorld(self, lat: np.ndarray) -> np.ndarray:
        return self.start_box + np.asarray(lat, dtype=np.float64) * self._d

    # ------------------------------------------------------------------ #
    # Forward evaluation = FFD3D (bspline3d.cc:633).  NOT divided by norm.
    # Points outside [-2,_x+1] x ... get zero displacement (the expanded
    # bbox keeps every template vertex inside).
    # ------------------------------------------------------------------ #
    def LocalDisplacement(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        N = pts.shape[0]
        lat = self.WorldToLattice(pts)
        l = np.floor(lat[:, 0]).astype(np.int64)
        m = np.floor(lat[:, 1]).astype(np.int64)
        n = np.floor(lat[:, 2]).astype(np.int64)
        Bx = _bspline_weights(lat[:, 0] - l)   # (N,4)
        By = _bspline_weights(lat[:, 1] - m)
        Bz = _bspline_weights(lat[:, 2] - n)

        # FFD3D returns 0 when the point is outside the lattice support.
        inside = (
            (lat[:, 0] >= -2) & (lat[:, 0] <= self._x + 1)
            & (lat[:, 1] >= -2) & (lat[:, 1] <= self._y + 1)
            & (lat[:, 2] >= -2) & (lat[:, 2] <= self._z + 1)
        )

        disp = np.zeros((N, 3), dtype=np.float64)
        NXp, NYp, NZp = self._x + 4, self._y + 4, self._z + 4
        for k in range(4):
            K = n + k - 1 + 2                       # padded array index
            Kc = np.clip(K, 0, NZp - 1)
            for j in range(4):
                J = m + j - 1 + 2
                Jc = np.clip(J, 0, NYp - 1)
                for i in range(4):
                    I = l + i - 1 + 2
                    Ic = np.clip(I, 0, NXp - 1)
                    w = Bz[:, k] * By[:, j] * Bx[:, i]   # (N,)
                    cp = self._data[Kc, Jc, Ic]          # (N,3)
                    disp += w[:, None] * cp
        disp[~inside] = 0.0
        return disp

    def Transform(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) + self.LocalDisplacement(pts)

    def Inverse(self, pts: np.ndarray, n_iter: int = 8) -> np.ndarray:
        """Inverse of Transform: find q s.t. q + LocalDisplacement(q) = pts.

        Faithful analogue of irtkTransformation::Inverse (a fixed-point / Newton
        iteration). The 1st-order inverse (q = pts - LocalDisplacement(pts)) is
        unstable for the symmetric snreg term; a few fixed-point steps converge
        and stop the error oscillation.
        """
        pts = np.asarray(pts, dtype=np.float64)
        q = pts - self.LocalDisplacement(pts)
        for _ in range(n_iter):
            q = pts - self.LocalDisplacement(q)
        return q

    # ------------------------------------------------------------------ #
    # Approximate3D — exact port of bspline3d.cc:1926.
    # pts: (N,3) FIXED world coords (original target points).
    # disp: (N,3) residual = source - currentTransform(target).
    # Updates self._data in place (+=).  Returns the residual error (mm).
    # ------------------------------------------------------------------ #
    def Approximate3D(self, pts: np.ndarray, disp: np.ndarray) -> float:
        pts = np.asarray(pts, dtype=np.float64)
        d = np.asarray(disp, dtype=np.float64).copy()
        N = pts.shape[0]
        NX, NY, NZ = self._x, self._y, self._z

        # (1) Subtract the displacement the current control points already produce.
        d -= self.LocalDisplacement(pts)

        # (2) Accumulators over the padded lattice.
        data = np.zeros((NZ + 4, NY + 4, NX + 4, 3), dtype=np.float64)  # numerator
        ds = np.zeros((NZ + 4, NY + 4, NX + 4), dtype=np.float64)       # denominator

        lat = self.WorldToLattice(pts)
        l = np.floor(lat[:, 0]).astype(np.int64)
        m = np.floor(lat[:, 1]).astype(np.int64)
        n = np.floor(lat[:, 2]).astype(np.int64)
        Bx = _bspline_weights(lat[:, 0] - l)   # (N,4)
        By = _bspline_weights(lat[:, 1] - m)
        Bz = _bspline_weights(lat[:, 2] - n)

        # B[index] = (4,4,4) tensor B_K * B_J * B_I  (axes k,j,i).
        # norm = Sum over ALL 64 of B_I^2 (FULL 4x4x4, NO bounds clip).
        B = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :])  # (N,4,4,4)
        norm = np.sum(B * B, axis=(1, 2, 3))          # (N,)
        # basis = B_I/norm ; data += B_I^2 * (d * basis) ; ds += B_I^2.
        # Combine: contribution to data = B_I^3 / norm * d ; to ds = B_I^2.
        with np.errstate(divide="ignore", invalid="ignore"):
            B2 = B * B                                 # (N,4,4,4) = B_I^2
            B3_over_norm = (B2 * B) / norm[:, None, None, None]   # (N,4,4,4)

        # Splat (clipped to padded range [-2 .. NX+1]).  Vectorise via flat
        # padded indices + np.add.at over the N*64 contributions.
        kk = (n[:, None] + np.arange(4)[None, :] - 1)   # (N,4) logical K
        mm = (m[:, None] + np.arange(4)[None, :] - 1)
        ll = (l[:, None] + np.arange(4)[None, :] - 1)
        Kg = kk[:, :, None, None] + 2                   # padded idx, broadcast (N,4,1,1)
        Jg = mm[:, None, :, None] + 2
        Ig = ll[:, None, None, :] + 2
        Kg = np.broadcast_to(Kg, (N, 4, 4, 4))
        Jg = np.broadcast_to(Jg, (N, 4, 4, 4))
        Ig = np.broadcast_to(Ig, (N, 4, 4, 4))
        valid = (
            (Kg >= 0) & (Kg < NZ + 4)
            & (Jg >= 0) & (Jg < NY + 4)
            & (Ig >= 0) & (Ig < NX + 4)
        )
        Kf = Kg[valid]
        Jf = Jg[valid]
        If = Ig[valid]
        wnum = B3_over_norm[valid]      # (M,)
        wden = B2[valid]                # (M,)
        contrib = wnum[:, None] * d[np.broadcast_to(
            np.arange(N)[:, None, None, None], (N, 4, 4, 4))[valid]]   # (M,3)
        np.add.at(data, (Kf, Jf, If), contrib)
        np.add.at(ds, (Kf, Jf, If), wden)

        # (3) (the "add back" of LocalDisplacement only affects d, which is no
        #      longer used; control update below is purely incremental.)
        # (4) Update control points: control += num/den where den>0.
        mask = ds > 0
        self._data[mask] += data[mask] / ds[mask][:, None]

        # Residual error (for logging) = mean |disp - LocalDisplacement| after update.
        resid = np.asarray(disp, dtype=np.float64) - self.LocalDisplacement(pts)
        return float(np.mean(np.linalg.norm(resid, axis=1)))


# --------------------------------------------------------------------------- #
# snreg — outer ICP wrapping Approximate3D (irtkSurfaceRegistration::Optimize).
# --------------------------------------------------------------------------- #
def snreg(
    target_pts: np.ndarray,
    source_pts: np.ndarray,
    *,
    spacing: float = 20.0,
    iterations: int = 100,
    epsilon: float = 0.01,
    symmetric: bool = True,
    verbose: bool = False,
) -> BSplineFFD:
    """Faithful IRTK snreg: FFD maps TARGET (template) -> SOURCE (subject).

    Build the FFD over the target bbox; build a cKDTree on the source once
    (the source never moves).  Each ICP iteration: transform target by the
    CURRENT ffd, snap to nearest source vertex, fit the residual with
    Approximate3D; optionally add the symmetric source->target term.
    """
    target_pts = np.asarray(target_pts, dtype=np.float64)
    source_pts = np.asarray(source_pts, dtype=np.float64)

    ffd = BSplineFFD(target_pts.min(0), target_pts.max(0), spacing=spacing)
    src_tree = cKDTree(source_pts)
    tgt_tree = cKDTree(target_pts) if symmetric else None

    last_error = 0.0
    # Keep the control grid that gave the LOWEST surface error so far. The ICP/MBA
    # error is not monotone — a finer control lattice in particular can overshoot
    # on an iteration and make the error jump UP. The old code returned whatever
    # grid was current when the stop test fired, i.e. the OVERSHOT one, so finer
    # spacings paradoxically fit worse. We snapshot the best grid and restore it.
    best_error = np.inf
    best_data = ffd._data.copy()
    for j in range(iterations):
        # --- forward correspondences: transform target by CURRENT ffd, snap to source ---
        moved = target_pts + ffd.LocalDisplacement(target_pts)   # Transform(p)
        dist, idx = src_tree.query(moved, k=1)
        tset = target_pts                          # ORIGINAL (untransformed) target pts
        sset = source_pts[idx]                     # matched closest source vertices
        error = float(dist.sum())
        n = len(target_pts)

        if symmetric:
            # map each source pt back through the inverse FFD, snap to target.
            inv = ffd.Inverse(source_pts)
            dist2, idx2 = tgt_tree.query(inv, k=1)
            tset = np.vstack([tset, target_pts[idx2]])
            sset = np.vstack([sset, source_pts])
            error += float(dist2.sum())
            n += len(source_pts)

        error /= n
        if verbose:
            print(f"  [snreg] Error = {error:.4f} at iteration = {j}")

        # This error reflects the CURRENT grid (the previous iteration's fit);
        # remember that grid if it is the best seen.
        if error < best_error:
            best_error = error
            best_data = ffd._data.copy()

        # --- IRTK stop test (AFTER error, BEFORE the next fit) ---
        if j > 0 and (last_error - error) < epsilon:
            break
        last_error = error

        # --- inner fit: residual = source - currentTransform(target) ---
        moved_t = tset + ffd.LocalDisplacement(tset)
        residual = sset - moved_t
        ffd.Approximate3D(tset, residual)          # updates ffd._data in place (+=)

    ffd._data = best_data  # restore the lowest-error grid (not the overshot one)
    return ffd


def fit_ffd(
    template_verts: np.ndarray,
    template_faces: np.ndarray,
    target_verts: np.ndarray,
    config: FFDConfig | None = None,
    *,
    control_spacing: float | None = None,
):
    """Deform the (already linearly-aligned) template onto the target surface.

    Runs IRTK snreg (target=template, source=subject), then returns the deformed
    template vertices (template topology preserved) plus a quality-metric info dict.

    Parameters
    ----------
    template_verts : (N,3) template vertices, after rigid+affine prealignment.
    template_faces : (M,3) template triangles (topology, preserved in the output).
    target_verts   : (P,3) subject region surface points.

    Returns
    -------
    fitted_verts : (N,3) float64, same ordering/topology as the input template.
    info : dict with mean/p95 symmetric surface distance.
    """
    cfg = config or FFDConfig()
    spacing = float(control_spacing) if control_spacing is not None else cfg.control_spacing

    tpl = np.asarray(template_verts, dtype=np.float64)
    tgt = np.asarray(target_verts, dtype=np.float64)

    # GPU snreg when a CUDA device is requested/available (numerically identical to
    # the numpy snreg — verified); otherwise the numpy path. ``device`` may be
    # "cuda" (default unless HEART_MESH_GPU=0), "cpu", or None/"auto".
    import os as _os

    want_gpu = str(getattr(cfg, "device", "cpu")).lower() in ("cuda", "gpu", "auto")
    if _os.environ.get("HEART_MESH_GPU", "1") == "0":
        want_gpu = False
    use_gpu = False
    if want_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

    if use_gpu:
        from .ffd_gpu import snreg_gpu

        gffd = snreg_gpu(
            tpl, tgt, spacing=spacing, iterations=cfg.iterations,
            epsilon=cfg.epsilon, symmetric=cfg.symmetric, device="cuda",
            verbose=cfg.verbose,
        )
        import torch

        fitted = (
            torch.as_tensor(tpl, dtype=torch.float64, device=gffd.device)
            + gffd.LocalDisplacement(tpl)
        ).cpu().numpy()
    else:
        ffd = snreg(
            tpl, tgt, spacing=spacing, iterations=cfg.iterations,
            epsilon=cfg.epsilon, symmetric=cfg.symmetric, verbose=cfg.verbose,
        )
        fitted = tpl + ffd.LocalDisplacement(tpl)

    # Quality: symmetric nearest-distance fitted<->target.
    d_ft = cKDTree(tgt).query(fitted)[0]
    d_tf = cKDTree(fitted).query(tgt)[0]
    d = np.concatenate([d_ft, d_tf])
    info = {
        "mean_surf_dist": float(d.mean()),
        "p95_surf_dist": float(np.percentile(d, 95)),
    }
    return fitted.astype(np.float64), info
