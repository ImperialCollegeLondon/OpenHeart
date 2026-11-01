"""VTK PolyData mesh I/O.

Reads/writes legacy VTK 4.2 PolyData files (the format the 4DSegment3.0 template
meshes use, written by IRTK / ``vtkPolyDataWriter`` with ``SetFileVersion(42)``).
Kept dependency-light: uses ``vtk`` for robust read/write of the exact same format
as the original pipeline so downstream IRTK feature tools stay compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


@dataclass
class Mesh:
    """A triangular surface mesh: vertices + triangle faces (+ optional scalars)."""

    verts: np.ndarray  # (N, 3) float
    faces: np.ndarray  # (M, 3) int  (triangles)
    point_scalars: dict | None = None  # name -> (N,) array

    @property
    def n_points(self) -> int:
        return len(self.verts)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    # ----------------------------------------------------------------- VTK <-> #
    def to_polydata(self) -> vtk.vtkPolyData:
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(self.verts, dtype=np.float64)))

        faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        cells = vtk.vtkCellArray()
        if hasattr(cells, "GetConnectivityArray"):  # VTK >= 9: offsets + connectivity in one go
            offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
            cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=1),
                          numpy_to_vtkIdTypeArray(faces.ravel(), deep=1))
        else:
            for tri in faces:
                cells.InsertNextCell(3)
                for i in tri:
                    cells.InsertCellPoint(int(i))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)

        if self.point_scalars:
            for name, arr in self.point_scalars.items():
                varr = numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float64), deep=1)
                varr.SetName(name)
                poly.GetPointData().AddArray(varr)
        return poly

    @classmethod
    def from_polydata(cls, poly: vtk.vtkPolyData) -> "Mesh":
        verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        cells = poly.GetPolys()
        if hasattr(cells, "GetConnectivityArray"):  # VTK >= 9
            offsets = vtk_to_numpy(cells.GetOffsetsArray())
            if len(offsets) > 1 and not np.all(np.diff(offsets) == 3):
                raise ValueError("expected a triangle mesh")
            faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
        else:  # legacy flat [3, a, b, c, 3, d, e, f, ...]
            faces = vtk_to_numpy(cells.GetData()).reshape(-1, 4)[:, 1:].astype(np.int64)
        scalars = {}
        pd = poly.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            name = pd.GetArrayName(i)
            scalars[name] = vtk_to_numpy(pd.GetArray(i))
        return cls(verts=verts, faces=faces, point_scalars=scalars or None)

    # --------------------------------------------------------------- file I/O #
    @classmethod
    def read(cls, path: str) -> "Mesh":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.ReadAllScalarsOn()  # keep every point scalar, not just the active one
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise IOError(f"Failed to read a non-empty mesh from {path}")
        # Triangulate to be safe (template meshes are already triangles).
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        return cls.from_polydata(tri.GetOutput())

    def write(self, path: str, *, file_version: int = 42) -> None:
        """Write a legacy VTK PolyData file (default v4.2, matching the pipeline)."""
        writer = vtk.vtkPolyDataWriter()
        try:
            writer.SetFileVersion(file_version)  # vtk >= 9
        except AttributeError:
            pass
        writer.SetFileName(str(path))
        writer.SetInputData(self.to_polydata())
        writer.Write()

    def write_txt(self, path: str, scalar: str | None = None) -> None:
        """Write the IRTK ``vtk2txt`` plain-text format.

        Each line is ``x y z`` (the vertex coordinates); when ``scalar`` names a
        per-vertex array it is appended as a 4th column. This matches the original
        4DSegment3.0 ``vtk2txt <mesh.vtk> <out.txt>`` output that the pipeline
        writes for ``lv_myo<fr>_wallthickness.txt`` etc. — verified against
        ``txt2cardiacvtk.cc`` (reads ``x y z [scalar]``, scalar last).
        """
        cols = [self.verts]
        if scalar is not None:
            if not self.point_scalars or scalar not in self.point_scalars:
                raise KeyError(f"mesh has no point scalar {scalar!r}")
            arr = np.asarray(self.point_scalars[scalar]).reshape(-1, 1)
            cols.append(arr)
        np.savetxt(str(path), np.hstack(cols))
