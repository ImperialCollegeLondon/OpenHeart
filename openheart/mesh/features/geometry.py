"""Per-vertex mesh features — pure-Python analogues of the IRTK ``cardiac*`` tools.

Because every fitted mesh shares the template topology, each feature is a length-N
per-vertex array whose entry *i* corresponds to the same anatomical location across
all subjects/phases — directly comparable across a cohort.

* wall_thickness            ~ cardiacwallthickness : LV-endo -> LV-epi distance
* wall_thickness_myocardium ~ cardiacwallthickness -myocardium : thickness on LVmyo
* signed_distance_to_template ~ cardiacenlargedistance / sevaluation : subject vs template
* mean_curvature            ~ cardiaccurvature / DiscreteCurvatureEstimator
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..io.mesh import Mesh


def wall_thickness(endo: Mesh, epi: Mesh) -> np.ndarray:
    """LV myocardial wall thickness at each endocardial vertex.

    For every endocardium vertex, the distance to the nearest epicardium point.
    This is the endo-side thickness field IRTK's ``cardiacwallthickness`` builds
    first (``array_target``). Returns a per-endo-vertex array (mm).

    To produce the myocardium mesh carrying this scalar — exactly what the
    original 4Dsegment pipeline writes (``W_LVmyo_<phase>.vtk``) — use
    :func:`wall_thickness_myocardium`.
    """
    tree = cKDTree(epi.verts)
    dist, _ = tree.query(endo.verts, k=1)
    return dist.astype(np.float64)


def wall_thickness_myocardium(endo: Mesh, epi: Mesh, myo: Mesh) -> Mesh:
    """LV wall thickness mapped onto the myocardium mesh, as IRTK does.

    Faithful port of ``cardiacwallthickness <endo> <epi> -myocardium <myo>`` from
    4Dsegment / IRTK. The original pipeline runs::

        cp F_LVmyo_<fr>.vtk W_LVmyo_<fr>.vtk
        cardiacwallthickness F_LVendo_<fr>.vtk F_LVepi_<fr>.vtk \
            -myocardium W_LVmyo_<fr>.vtk

    Algorithm (matching the C++ in ``cardiacwallthickness.cc``):

    1. ``array_target``: each *endo* vertex's distance to the nearest *epi* point.
    2. ``array_source``: each *epi* vertex's distance to the nearest *endo* point.
    3. For each *myo* vertex, find its nearest point on endo (dist ``rerror``) and
       on epi (dist ``rerror2``); assign the thickness value of whichever surface
       is closer — ``array_source[id2]`` if epi is nearer, else ``array_target[id]``.

    Because the mapping is by nearest-point lookup, endo / epi / myo need *not*
    share topology or vertex counts (they don't in this template). Returns a copy
    of ``myo`` carrying a ``"WallThickness"`` per-vertex scalar (mm).

    Verified bit-for-bit against ``cardiacwallthickness.cc`` (4DSegment3.0 /
    template_wenjia atlas, ED+ES) except at *exact-distance ties*: when a myo
    vertex is equidistant from two endo (or two epi) vertices with differing
    thickness, ``cKDTree`` and IRTK's ``irtkLocator`` may pick different
    neighbours. Such ties are vanishingly rare (ED: 0/22043; ES: 3/13646 ≈
    0.02 %, max ~0.13 mm — below segmentation voxel noise) and the choice between
    two equidistant neighbours is arbitrary, so this is left as a known tolerance
    rather than reproducing VTK's implementation-defined tie ordering.
    """
    endo_tree = cKDTree(endo.verts)
    epi_tree = cKDTree(epi.verts)

    # Step 1 & 2: per-surface thickness fields.
    array_target, _ = epi_tree.query(endo.verts, k=1)   # endo -> nearest epi (mm)
    array_source, _ = endo_tree.query(epi.verts, k=1)   # epi  -> nearest endo (mm)

    # Step 3: for each myo vertex pick the thickness of the nearer surface.
    rerror, id_endo = endo_tree.query(myo.verts, k=1)   # myo -> endo
    rerror2, id_epi = epi_tree.query(myo.verts, k=1)    # myo -> epi
    epi_closer = rerror2 < rerror
    wt = np.where(epi_closer, array_source[id_epi], array_target[id_endo])

    scalars = dict(myo.point_scalars) if myo.point_scalars else {}
    scalars["WallThickness"] = wt.astype(np.float64)
    return Mesh(verts=myo.verts.copy(), faces=myo.faces.copy(), point_scalars=scalars)


def signed_distance_to_template(subject: Mesh, template: Mesh) -> np.ndarray:
    """Signed distance from each subject vertex to the template surface.

    Sign is positive when the subject vertex lies *outside* the template surface
    (along the template outward normal), negative inside — matching the intent of
    IRTK ``sevaluation -signed`` / ``cardiacenlargedistance``. Requires shared
    topology (subject and template have the same vertex ordering), which holds for
    fitted meshes, so we can also use the direct per-vertex displacement sign.
    """
    if subject.n_points == template.n_points:
        # Shared topology: signed along the template vertex normal.
        normals = _vertex_normals(template)
        disp = subject.verts - template.verts
        signed = np.einsum("ij,ij->i", disp, normals)
        return signed.astype(np.float64)
    # Fallback: unsigned nearest-surface distance with normal-based sign.
    tree = cKDTree(template.verts)
    dist, idx = tree.query(subject.verts, k=1)
    normals = _vertex_normals(template)
    disp = subject.verts - template.verts[idx]
    sign = np.sign(np.einsum("ij,ij->i", disp, normals[idx]))
    return (dist * sign).astype(np.float64)


def _vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted vertex normals."""
    v, f = mesh.verts, mesh.faces
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # face normals*2area
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vn / norm


def mean_curvature(mesh: Mesh, smooth_iters: int = 0) -> np.ndarray:
    """Discrete mean curvature per vertex (cotangent Laplace-Beltrami).

    Analogue of IRTK ``DiscreteCurvatureEstimator`` + ``cardiaccurvature``. Uses the
    cotangent-weighted Laplacian; the mean-curvature magnitude is half the norm of
    the Laplace-Beltrami operator applied to the vertex positions, signed by the
    surface normal direction.
    """
    v, f = mesh.verts, mesh.faces
    n = len(v)

    # Cotangent weights per triangle corner.
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    e0 = v[i2] - v[i1]
    e1 = v[i0] - v[i2]
    e2 = v[i1] - v[i0]

    def _cot(a, b):
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        cross[cross < 1e-12] = 1e-12
        return np.einsum("ij,ij->i", a, b) / cross

    cot0 = _cot(-e1, e2)
    cot1 = _cot(-e2, e0)
    cot2 = _cot(-e0, e1)

    L = np.zeros((n, 3))
    area = np.zeros(n)
    for (a, b, w) in [
        (i1, i2, cot0),
        (i2, i0, cot1),
        (i0, i1, cot2),
    ]:
        diff_ab = v[a] - v[b]
        np.add.at(L, a, -(w[:, None]) * diff_ab)
        np.add.at(L, b, (w[:, None]) * diff_ab)

    # Vertex (barycentric) areas for normalisation.
    tri_area = 0.5 * np.linalg.norm(np.cross(e2, -e1), axis=1)
    for idx in (i0, i1, i2):
        np.add.at(area, idx, tri_area / 3.0)
    area[area < 1e-12] = 1e-12

    Hvec = L / (2.0 * area[:, None])
    H = 0.5 * np.linalg.norm(Hvec, axis=1)
    # Sign by alignment with outward normal (concave vs convex).
    normals = _vertex_normals(mesh)
    sign = np.sign(np.einsum("ij,ij->i", Hvec, normals))
    H = H * np.where(sign == 0, 1.0, sign)

    for _ in range(smooth_iters):
        H = _smooth_scalar(H, f, n)
    return H.astype(np.float64)


def _smooth_scalar(scalar: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """One iteration of umbrella (uniform-Laplacian) smoothing of a scalar field."""
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for a, b in [(faces[:, 0], faces[:, 1]), (faces[:, 1], faces[:, 2]), (faces[:, 2], faces[:, 0])]:
        np.add.at(acc, a, scalar[b]); np.add.at(cnt, a, 1)
        np.add.at(acc, b, scalar[a]); np.add.at(cnt, b, 1)
    cnt[cnt == 0] = 1
    return 0.5 * scalar + 0.5 * (acc / cnt)
