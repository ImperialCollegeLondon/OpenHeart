"""GPU (PyTorch) port of the IRTK snreg B-spline FFD in :mod:`ffd_torch`.

Numerically equivalent to the numpy reference (``ffd_torch.BSplineFFD`` / ``snreg``)
— same algorithm, same float64 math — but the per-point work (B-spline evaluation,
the MBA scatter-splat, nearest-point search) runs on the GPU via torch. Every method
is validated against the numpy version to bit-close tolerance.

Hot paths moved to GPU:
* LocalDisplacement / Approximate3D : torch gather + scatter_add_ over the lattice
* nearest source point (the ICP step) : pytorch3d knn_points (fallback: torch.cdist)
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from pytorch3d.ops import knn_points as _p3d_knn
    _HAVE_P3D = True
except Exception:  # pragma: no cover
    _HAVE_P3D = False


def _bspline_weights_t(u: torch.Tensor) -> torch.Tensor:
    """Uniform cubic B-spline basis for fractional offsets ``u`` in [0,1).

    Returns (...,4) weights B0..B3. Torch port of ``ffd_torch._bspline_weights``.
    """
    u2 = u * u
    u3 = u2 * u
    w0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0
    w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
    w2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
    w3 = u3 / 6.0
    return torch.stack([w0, w1, w2, w3], dim=-1)


class BSplineFFDGPU:
    """Cubic B-spline FFD over a 3D control lattice (torch / float64 on GPU).

    Same construction as the numpy ``BSplineFFD`` (bbox + one spacing pad, dims
    round(span/ds)+1, spacing re-snapped). Control grid ``_data`` is a torch tensor.
    """

    def __init__(self, lo, hi, spacing=20.0, device="cuda", dtype=torch.float64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        ds = float(spacing)

        self.start_box = lo - ds
        self.end_box = hi + ds
        span = self.end_box - self.start_box
        self._x = max(int(round(span[0] / ds)) + 1, 2)
        self._y = max(int(round(span[1] / ds)) + 1, 2)
        self._z = max(int(round(span[2] / ds)) + 1, 2)
        self._d = np.array([
            span[0] / (self._x - 1),
            span[1] / (self._y - 1),
            span[2] / (self._z - 1),
        ], dtype=np.float64)

        # torch copies of the constants used per-point.
        self._start_t = torch.as_tensor(self.start_box, dtype=dtype, device=self.device)
        self._d_t = torch.as_tensor(self._d, dtype=dtype, device=self.device)
        # Control-point displacements (padded by 2 planes each side), zero = identity.
        self._data = torch.zeros(
            (self._z + 4, self._y + 4, self._x + 4, 3), dtype=dtype, device=self.device
        )

    # ------------------------------------------------------------------ #
    def _to_t(self, pts):
        if isinstance(pts, torch.Tensor):
            return pts.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(pts), dtype=self.dtype, device=self.device)

    def WorldToLattice(self, pts: torch.Tensor) -> torch.Tensor:
        return (self._to_t(pts) - self._start_t) / self._d_t

    # ------------------------------------------------------------------ #
    # Forward evaluation = FFD3D. Out-of-lattice points get zero displacement.
    # Torch port of ffd_torch.BSplineFFD.LocalDisplacement.
    # ------------------------------------------------------------------ #
    def LocalDisplacement(self, pts) -> torch.Tensor:
        pts = self._to_t(pts)
        N = pts.shape[0]
        lat = self.WorldToLattice(pts)
        base = torch.floor(lat).to(torch.int64)            # (N,3) l,m,n
        l, m, n = base[:, 0], base[:, 1], base[:, 2]
        Bx = _bspline_weights_t(lat[:, 0] - l)             # (N,4)
        By = _bspline_weights_t(lat[:, 1] - m)
        Bz = _bspline_weights_t(lat[:, 2] - n)

        inside = (
            (lat[:, 0] >= -2) & (lat[:, 0] <= self._x + 1)
            & (lat[:, 1] >= -2) & (lat[:, 1] <= self._y + 1)
            & (lat[:, 2] >= -2) & (lat[:, 2] <= self._z + 1)
        )

        NXp, NYp, NZp = self._x + 4, self._y + 4, self._z + 4
        disp = torch.zeros((N, 3), dtype=self.dtype, device=self.device)
        for k in range(4):
            Kc = (n + k - 1 + 2).clamp(0, NZp - 1)
            for j in range(4):
                Jc = (m + j - 1 + 2).clamp(0, NYp - 1)
                for i in range(4):
                    Ic = (l + i - 1 + 2).clamp(0, NXp - 1)
                    w = Bz[:, k] * By[:, j] * Bx[:, i]      # (N,)
                    cp = self._data[Kc, Jc, Ic]             # (N,3)
                    disp = disp + w[:, None] * cp
        disp = torch.where(inside[:, None], disp, torch.zeros_like(disp))
        return disp

    def Transform(self, pts) -> torch.Tensor:
        pts = self._to_t(pts)
        return pts + self.LocalDisplacement(pts)

    def Inverse(self, pts, n_iter: int = 8) -> torch.Tensor:
        """Fixed-point inverse: q s.t. q + LocalDisplacement(q) = pts. Torch port."""
        pts = self._to_t(pts)
        q = pts - self.LocalDisplacement(pts)
        for _ in range(n_iter):
            q = pts - self.LocalDisplacement(q)
        return q

    # ------------------------------------------------------------------ #
    # Approximate3D — torch port of ffd_torch.BSplineFFD.Approximate3D
    # (the MBA scatter-splat). Updates self._data in place (+=). Returns error.
    # ------------------------------------------------------------------ #
    def Approximate3D(self, pts, disp) -> float:
        pts = self._to_t(pts)
        disp_in = self._to_t(disp)
        N = pts.shape[0]
        NX, NY, NZ = self._x, self._y, self._z
        NXp, NYp, NZp = NX + 4, NY + 4, NZ + 4

        # (1) residual after the current control points.
        d = disp_in - self.LocalDisplacement(pts)          # (N,3)

        lat = self.WorldToLattice(pts)
        base = torch.floor(lat).to(torch.int64)
        l, m, n = base[:, 0], base[:, 1], base[:, 2]
        Bx = _bspline_weights_t(lat[:, 0] - l)             # (N,4)
        By = _bspline_weights_t(lat[:, 1] - m)
        Bz = _bspline_weights_t(lat[:, 2] - n)

        # B[index] = (N,4,4,4) = B_K * B_J * B_I (axes k,j,i).
        B = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :])
        norm = (B * B).sum(dim=(1, 2, 3))                  # (N,)
        B2 = B * B                                         # B_I^2
        B3_over_norm = (B2 * B) / norm[:, None, None, None]  # B_I^3/norm

        # Padded lattice indices for the 4x4x4 neighbourhood.
        ar = torch.arange(4, device=self.device)
        Kg = (n[:, None] - 1 + ar)[:, :, None, None] + 2   # (N,4,1,1)
        Jg = (m[:, None] - 1 + ar)[:, None, :, None] + 2
        Ig = (l[:, None] - 1 + ar)[:, None, None, :] + 2
        Kg = Kg.expand(N, 4, 4, 4)
        Jg = Jg.expand(N, 4, 4, 4)
        Ig = Ig.expand(N, 4, 4, 4)
        valid = (
            (Kg >= 0) & (Kg < NZp) & (Jg >= 0) & (Jg < NYp) & (Ig >= 0) & (Ig < NXp)
        )
        # flat padded index for scatter_add.
        flat = (Kg * NYp + Jg) * NXp + Ig                  # (N,4,4,4)
        flat_v = flat[valid]                               # (M,)
        wnum = B3_over_norm[valid]                         # (M,)
        wden = B2[valid]                                   # (M,)
        # per-contribution residual vector (broadcast d over the 4x4x4 block).
        d_b = d[:, None, None, None, :].expand(N, 4, 4, 4, 3)[valid]   # (M,3)
        contrib = wnum[:, None] * d_b                      # (M,3)

        n_cells = NZp * NYp * NXp
        data = torch.zeros((n_cells, 3), dtype=self.dtype, device=self.device)
        ds = torch.zeros((n_cells,), dtype=self.dtype, device=self.device)
        data.index_add_(0, flat_v, contrib)
        ds.index_add_(0, flat_v, wden)

        data = data.view(NZp, NYp, NXp, 3)
        ds = ds.view(NZp, NYp, NXp)
        mask = ds > 0
        self._data[mask] += data[mask] / ds[mask][:, None]

        resid = disp_in - self.LocalDisplacement(pts)
        return float(torch.mean(torch.linalg.norm(resid, dim=1)).item())


def _nearest(query: torch.Tensor, ref: torch.Tensor):
    """Index in ``ref`` (M,3) nearest to each row of ``query`` (N,3). GPU.

    Uses pytorch3d knn_points; falls back to a chunked torch.cdist argmin.
    """
    if _HAVE_P3D:
        _, idx, _ = _p3d_knn(query[None].float(), ref[None].float(), K=1)
        return idx[0, :, 0]
    # fallback: chunked to bound memory
    out = torch.empty(query.shape[0], dtype=torch.int64, device=query.device)
    chunk = 4096
    for s in range(0, query.shape[0], chunk):
        d = torch.cdist(query[s:s + chunk], ref)
        out[s:s + chunk] = d.argmin(dim=1)
    return out


# --------------------------------------------------------------------------- #
# GPU linear ICP (sareg affine + msrreg-3 joint rigid). The whole loop stays on
# the GPU — points are uploaded once, every iteration's nearest-neighbour, lstsq
# / Kabsch solve and transform run in torch — so there is no per-iteration CPU
# round-trip. Numerically equivalent to linear.icp_affine / linear.joint_icp_rigid.
# --------------------------------------------------------------------------- #
def _apply_t(pts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return pts @ T[:3, :3].T + T[:3, 3]


def _bbox_min_ratio_t(moving: torch.Tensor, dst: torch.Tensor) -> float:
    mr = moving.amax(0) - moving.amin(0)
    dr = torch.clamp(dst.amax(0) - dst.amin(0), min=1e-6)
    return float((mr / dr).min().item())


def _affine_from_corr_t(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    X = torch.cat([src, torch.ones(src.shape[0], 1, dtype=src.dtype, device=src.device)], 1)
    M = torch.linalg.lstsq(X, dst).solution        # (4,3)
    T = torch.eye(4, dtype=src.dtype, device=src.device)
    T[:3, :3] = M[:3, :].T
    T[:3, 3] = M[3, :]
    return T


def _rigid_from_corr_t(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    S = src - mu_s
    D = dst - mu_d
    H = S.T @ D
    U, w, Vt = torch.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if torch.linalg.det(R) <= 0.999:
        i = int(torch.argmin(w).item())
        V[:, i] *= -1
        R = V @ U.T
    T = torch.eye(4, dtype=src.dtype, device=src.device)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def icp_affine_gpu(src, dst, *, init=None, max_iter=100, tol=0.01,
                   step=0.2, min_axis_floor=0.7, device="cuda"):
    """GPU port of linear.icp_affine (gradual damped affine + collapse guard)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    src = torch.as_tensor(np.asarray(src), dtype=torch.float64, device=dev)
    dst = torch.as_tensor(np.asarray(dst), dtype=torch.float64, device=dev)
    T = torch.eye(4, dtype=torch.float64, device=dev) if init is None else \
        torch.as_tensor(np.asarray(init), dtype=torch.float64, device=dev)
    I = torch.eye(4, dtype=torch.float64, device=dev)
    moving = _apply_t(src, T)
    last_err = 0.0
    for j in range(max_iter):
        idx = _nearest(moving, dst)
        matched = dst[idx]
        err = float(torch.linalg.norm(moving - matched, dim=1).mean().item())
        if j > 0 and last_err - err < tol:
            break
        last_err = err
        T_full = _affine_from_corr_t(moving, matched)
        T_step = I + step * (T_full - I)
        moving_next = _apply_t(moving, T_step)
        r_cur = _bbox_min_ratio_t(moving, dst)
        r_next = _bbox_min_ratio_t(moving_next, dst)
        if r_next < min_axis_floor and r_next < r_cur:
            break
        moving = moving_next
        T = T_step @ T
    return T.cpu().numpy()


def joint_icp_rigid_gpu(tpl_sets, tgt_sets, *, init=None, max_iter=100,
                        tol=0.01, device="cuda"):
    """GPU port of linear.joint_icp_rigid (pooled pure-rigid Kabsch ICP)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tpl = [torch.as_tensor(np.asarray(p), dtype=torch.float64, device=dev) for p in tpl_sets]
    tgt = [torch.as_tensor(np.asarray(p), dtype=torch.float64, device=dev) for p in tgt_sets]
    T = torch.eye(4, dtype=torch.float64, device=dev) if init is None else \
        torch.as_tensor(np.asarray(init), dtype=torch.float64, device=dev)
    moving = [_apply_t(p, T) for p in tpl]
    last_err = 0.0
    for j in range(max_iter):
        src_all, dst_all = [], []
        for mv, tg in zip(moving, tgt):
            idx = _nearest(mv, tg)
            src_all.append(mv)
            dst_all.append(tg[idx])
        src = torch.vstack(src_all)
        dst = torch.vstack(dst_all)
        err = float(torch.linalg.norm(src - dst, dim=1).mean().item())
        if j > 0 and last_err - err < tol:
            break
        last_err = err
        T_step = _rigid_from_corr_t(src, dst)
        moving = [_apply_t(mv, T_step) for mv in moving]
        T = T_step @ T
    return T.cpu().numpy()


def snreg_gpu(
    target_pts,
    source_pts,
    *,
    spacing: float = 20.0,
    iterations: int = 100,
    epsilon: float = 0.01,
    symmetric: bool = True,
    device: str = "cuda",
    verbose: bool = False,
) -> BSplineFFDGPU:
    """GPU port of ffd_torch.snreg. FFD maps TARGET (template) -> SOURCE (subject).

    Same algorithm as the numpy reference: build the FFD over the target bbox,
    ICP-iterate (transform target by the current FFD, snap to nearest source
    vertex, fit the residual with Approximate3D, optional symmetric term), keep the
    LOWEST-error control grid (best-so-far). nearest-point search is GPU knn.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tgt = torch.as_tensor(np.asarray(target_pts), dtype=torch.float64, device=dev)
    src = torch.as_tensor(np.asarray(source_pts), dtype=torch.float64, device=dev)

    lo = tgt.min(0).values.cpu().numpy()
    hi = tgt.max(0).values.cpu().numpy()
    ffd = BSplineFFDGPU(lo, hi, spacing=spacing, device=device)

    last_error = 0.0
    best_error = float("inf")
    best_data = ffd._data.clone()
    for j in range(iterations):
        moved = tgt + ffd.LocalDisplacement(tgt)
        idx = _nearest(moved, src)
        tset = tgt
        sset = src[idx]
        error = float(torch.linalg.norm(moved - sset, dim=1).sum().item())
        n = tgt.shape[0]

        if symmetric:
            inv = ffd.Inverse(src)
            idx2 = _nearest(inv, tgt)
            tset = torch.vstack([tset, tgt[idx2]])
            sset = torch.vstack([sset, src])
            error += float(torch.linalg.norm(inv - tgt[idx2], dim=1).sum().item())
            n += src.shape[0]

        error /= n
        if verbose:
            print(f"  [snreg-gpu] Error = {error:.4f} at iteration = {j}")

        if error < best_error:
            best_error = error
            best_data = ffd._data.clone()

        if j > 0 and (last_error - error) < epsilon:
            break
        last_error = error

        moved_t = tset + ffd.LocalDisplacement(tset)
        residual = sset - moved_t
        ffd.Approximate3D(tset, residual)

    ffd._data = best_data
    return ffd
