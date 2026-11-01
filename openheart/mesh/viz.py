"""Lightweight visual-fit QC: render fitted meshes over the target surface.

The acceptance criterion for this package is *visual* fit (fitted template surface
sits on the segmentation boundary) plus preserved topology. These helpers produce
off-screen PNGs so fit quality can be eyeballed without a GUI — handy on headless
servers/HPC.

Requires ``pyvista`` (``pip install openheart[viz]``).
"""

from __future__ import annotations

import numpy as np

from .io.mesh import Mesh
from .io.segmentation import Volume, extract_surface


def _pv_polydata(mesh: Mesh):
    import pyvista as pv

    faces = np.c_[np.full(len(mesh.faces), 3), mesh.faces].ravel()
    return pv.PolyData(mesh.verts, faces)


def overlay_png(
    fitted: Mesh,
    target_verts: np.ndarray,
    out_png: str,
    *,
    scalars: np.ndarray | None = None,
    title: str | None = None,
):
    """Save a PNG of the fitted mesh (solid) over the target surface points (red).

    Parameters
    ----------
    fitted : fitted template mesh.
    target_verts : (P,3) target surface points (subject segmentation surface).
    scalars : optional per-vertex scalar to colour the fitted mesh (e.g. wall thickness).
    """
    import pyvista as pv

    pv.global_theme.allow_empty_mesh = True
    pl = pv.Plotter(off_screen=True, window_size=(900, 800))
    poly = _pv_polydata(fitted)
    if scalars is not None:
        poly["scalar"] = np.asarray(scalars)
        pl.add_mesh(poly, scalars="scalar", cmap="viridis", opacity=1.0, show_edges=False)
    else:
        pl.add_mesh(poly, color="lightblue", opacity=0.6, show_edges=False)
    pl.add_points(
        np.asarray(target_verts), color="red", point_size=2.0, render_points_as_spheres=False
    )
    if title:
        pl.add_text(title, font_size=10)
    pl.view_isometric()
    pl.screenshot(out_png)
    pl.close()
    return out_png


def overlay_from_segmentation(
    fitted: Mesh, seg_path: str, region: str, out_png: str, *, labels=None, **kw
):
    """Convenience: extract the target surface from a segmentation and overlay.

    ``labels`` sets the segmentation's label convention (see ``LabelMap``); defaults
    to the 4DSegment convention (1=LV, 2=myo, 3=RV).
    """
    vol = Volume.load(seg_path, labels=labels)
    tv, _ = extract_surface(vol, region)
    return overlay_png(fitted, tv, out_png, **kw)
