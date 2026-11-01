# openheart.mesh — template mesh fitting

GPU cardiac **template mesh fitting** — a pure-Python reimplementation of the
**mesh-fitting stage of [4DSegment3.0](https://github.com/ImperialCollegeLondon/4DSegment3.0)**
(Imperial College London). Fit a fixed-topology cardiac template (LV endocardium /
epicardium / myocardium + RV) to a segmentation label map, **preserving point
correspondence** across subjects and phases so the results are directly comparable
vertex-by-vertex.

> **This is a faithful port of 4DSegment3.0's mesh fitting, not a new method.**
> The registration logic, template, and feature definitions come from 4DSegment3.0
> ([ImperialCollegeLondon/4DSegment3.0](https://github.com/ImperialCollegeLondon/4DSegment3.0),
> itself derived from [4DSegment2.0](https://github.com/ImperialCollegeLondon/4DSegment2.0)
> and the original [j-duan/4Dsegment](https://github.com/j-duan/4Dsegment), built on
> [IRTK](https://github.com/BioMedIA/IRTK)/[MIRTK](https://github.com/BioMedIA/MIRTK)).
> This package re-implements **only the mesh-fitting + feature-extraction stage** in
> GPU PyTorch. See [Relation to 4DSegment3.0](#relation-to-4dsegment30) for exactly
> what is the same and what differs.

The original pipeline did this with compiled IRTK/MIRTK C++ binaries
(`prreg` / `msrreg` / `sareg` / `snreg`) that only existed inside a ~10 GB Docker
image. This package replaces that stage with GPU-accelerated PyTorch so it can be
`pip`-installed and run anywhere — no Docker, no IRTK.

| IRTK / 4DSegment stage | here |
|---|---|
| `binarize` + `mcubes` | `skimage.measure.marching_cubes` |
| landmark extraction (`mesh_ed_es.py`) | numpy port (`landmarks.py`) |
| `prreg` (rigid landmarks) | Kabsch/Umeyama closed form |
| `msrreg` (rigid surface) | ICP rigid+scale |
| `sareg` (affine surface) | ICP affine |
| `snreg` (non-rigid B-spline FFD) | **GPU cubic B-spline FFD** (PyTorch) |
| `cardiacwallthickness` / `cardiaccurvature` / `sevaluation` | `features/` (pure numpy) |

## What you get

* **Topology preserved.** Every fitted mesh has the template's exact vertex count
  and connectivity (LVendo 9207, LVepi 12836, LVmyo 22043, RV 14622 points), so
  vertex *i* is the same anatomical location in every subject/phase → cross-case /
  cross-time analysis works out of the box. *(See [Relation to 4DSegment3.0](#relation-to-4dsegment30)
  — the original's released surface-mapping branch did **not** guarantee this.)*
* **GPU non-rigid fit.** Sub-millimetre surface fit on a V100 in seconds.
* **Bundled template.** The heart template ships inside the package — clone & go.
* **Per-vertex features.** Wall thickness, signed distance to template, curvature.
* **Any label convention.** Point it at your own labels via a label spec or
  `LabelMap` — no need to relabel your segmentations first. See
  [Configuring labels](#configuring-labels).

## Relation to 4DSegment3.0

This module re-implements **one stage** of 4DSegment3.0 — the mesh fitting and
per-vertex feature extraction (`scripts/mesh_ed_es.py` + `scripts/utils/imaging.py`
in the original). It does **not** do the deep-learning segmentation, ED/ES detection,
or cine-tracking stages; it starts from a segmentation label map and ends at fitted
meshes + features.

**What is the same (faithful to 4DSegment3.0):**

* **Same template.** The bundled atlas is the original `template_wenjia` mesh set,
  byte-for-byte the same vertex counts per region/phase.
* **Same registration cascade.** landmark rigid → joint surface rigid (`msrreg`) →
  per-region affine (`sareg`) → non-rigid B-spline FFD (`snreg`), with the same
  control-point spacing and the same ED→ES initialisation logic.
* **Same feature definitions.** Wall thickness, signed distance and curvature follow
  the IRTK `cardiac*` tools. Wall thickness in particular is cross-checked
  **bit-for-bit** against `cardiacwallthickness.cc` (see below).

**What differs (by design):**

| | 4DSegment3.0 | here |
|---|---|---|
| Implementation | IRTK/MIRTK C++ binaries in a ~10 GB Docker image | pure-Python + **GPU PyTorch**, `pip install` |
| `snreg` solver | CPU B-spline FFD | GPU cubic B-spline FFD (autograd), ~6.9× faster, equivalent fit |
| **Output topology** | the released `transform-points` branch could emit meshes whose vertex count/connectivity **varied per subject** | **always the template's exact topology** → vertex *i* is the same anatomy in every case/phase, so cohort analysis needs no remeshing |
| Label convention | fixed `1=LV,2=myo,3=RV` | configurable label spec / `LabelMap` |
| Wall-thickness tie-break | VTK locator | `cKDTree` — identical except at exact-distance ties (≤0.02 % of myo vertices; [documented](#features)) |

The key **logical** difference is the topology guarantee: 4DSegment3.0 deforms the
template into subject space (so its fitted meshes keep template topology), but its
*released* surface-mapping branch did not guarantee shared connectivity across
subjects. Here, every region is fit by deforming the bundled template, so point
correspondence is guaranteed by construction — the property the downstream
cross-subject / cross-time analyses rely on.

## Install

With the mesh extra (torch, scikit-image, vtk):

```bash
pip install -e ".[mesh]"      # from a clone of OpenHeart
```

### PyTorch3D (recommended, for fast/smooth fitting)

PyTorch3D gives a faster chamfer loss + mesh regularisers (Laplacian / edge /
normal) for smoother, more anatomically plausible meshes. It must be built to
match your `torch` + CUDA. With a CUDA toolchain available:

```bash
# matched compiler + nvcc (example via conda)
conda install -c "nvidia/label/cuda-12.1.0" -c conda-forge \
    cuda-nvcc=12.1 cuda-libraries-dev=12.1 gxx_linux-64=11 ninja
export CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

**Without PyTorch3D the package still works** — it falls back to a self-contained
chamfer distance (a little slower, no mesh regularisers).

## Usage

### Python

```python
from openheart.mesh import fit_subject

results = fit_subject(
    "seg_lvsa_SR_ED.nii.gz",
    "seg_lvsa_SR_ES.nii.gz",   # optional
    out_dir="out/",            # writes F_<region>_<phase>.vtk
)

ed = results["ED"]
mesh = ed.meshes["LVepi"]
print(mesh.n_points)                 # 12836 — same for every subject
print(ed.metrics["LVepi"])           # {'mean_surf_dist': 0.6, ...}
```

> The examples above assume the default 4DSegment label convention
> (`1=LV, 2=myo, 3=RV`). For any other convention, pass `labels=...`
> (Python) or `--labels ...` (CLI) — see [Configuring labels](#configuring-labels).

### Command line

```bash
openheart fit --ed seg_lvsa_SR_ED.nii.gz --es seg_lvsa_SR_ES.nii.gz \
              --out out/ --features

# non-default label convention (e.g. 1=RV, 2=myo, 3=LV):
openheart fit --ed seg_ED.nii.gz --out out/ --labels lv=3,myo=2,rv=1
```

### Features

```python
from openheart.mesh import features
wt = features.wall_thickness(ed.meshes["LVendo"], ed.meshes["LVepi"])  # per-endo-vertex mm

# Faithful 4Dsegment output: thickness scalar carried on the LVmyo mesh, exactly
# as `cardiacwallthickness F_LVendo F_LVepi -myocardium W_LVmyo.vtk` produces.
myo = features.wall_thickness_myocardium(
    ed.meshes["LVendo"], ed.meshes["LVepi"], ed.meshes["LVmyo"]
)
myo.write("W_LVmyo_ED.vtk")  # VTK mesh with a "WallThickness" point scalar
```

With `--features`, the CLI writes `W_LVmyo_<phase>.vtk` (thickness as the
`WallThickness` point scalar) alongside `mesh_features/lv_myo<phase>_wallthickness.txt`.

## Input format

Segmentation NIfTI. The default label convention is **1 = LV cavity, 2 = myocardium,
3 = RV** (the 4DSegment convention). This package starts from segmentations — it
does **not** do the deep-learning segmentation step.

### Configuring labels

If your segmentation uses a different convention, tell the fitter — no external
relabelling needed. For example, with LV and RV swapped (`1 = RV, 2 = myo, 3 = LV`):

```bash
openheart fit --ed seg_ED.nii.gz --out out/ --labels lv=3,myo=2,rv=1
```

```python
from openheart.mesh import fit_subject, LabelMap

# an explicit mapping (any of these forms work):
fit_subject("seg_ED.nii.gz", out_dir="out/", labels="lv=3,myo=2,rv=1")
fit_subject("seg_ED.nii.gz", out_dir="out/", labels=LabelMap(lv=3, myo=2, rv=1))
```

The default is `4dseg` (`1=LV,2=myo,3=RV`). The label spec also accepts a dict
(`{"lv":3,"myo":2,"rv":1}`) or positional `"3,2,1"`. Getting this wrong fits the
LV template to the RV cavity and vice-versa, so set it to match your data.

## The template

Every fitted mesh is a deformation of a **template** (atlas) mesh, which is what
gives the output its fixed topology. The package ships one — the original
4DSegment3.0 `template_wenjia` atlas — under `openheart/mesh/assets/template/`,
so a fresh clone works with no download.

A template is a directory with this exact layout:

```
template/
├── segmentation_ED.nii.gz          # ED label map; template landmarks are derived from it at runtime
└── vtks/
    ├── LVendo_ED.vtk   LVendo_ES.vtk
    ├── LVepi_ED.vtk    LVepi_ES.vtk
    ├── LVmyo_ED.vtk    LVmyo_ES.vtk
    └── RV_ED.vtk       RV_ES.vtk
```

i.e. one VTK PolyData mesh per **region** × **phase** (`LVendo`, `LVepi`, `LVmyo`,
`RV` × `ED`, `ES`) plus an ED segmentation. The meshes are legacy VTK 4.2 PolyData
(as written by IRTK / `vtkPolyDataWriter` with `SetFileVersion(42)`).

**Vertex counts are a property of the template, not the code.** Nothing in the
fitter hard-codes a vertex count — the bundled atlas happens to have LVendo 9207 /
LVepi 12836 / LVmyo 22043 / RV 14622 (ED), but a different template with **different
per-region vertex counts is fine**; every output simply inherits *that* template's
topology. (The wall-thickness mapping, for example, uses nearest-point lookup
precisely so endo/epi/myo need not share vertex counts.) What matters is that within
one template the regions are mutually consistent and the ED segmentation matches the
ED meshes.

**Swapping in your own template:** pass its folder with `template=` (`fit_subject`,
`fit_phase`, `oh.FitMesh`) or `--template` (`openheart fit`, `openheart run`). Its
`segmentation_ED.nii.gz` uses labels `1=LV, 2=myo, 3=RV`. The unused
`landmarks3_ED.vtk` in the bundled `vtks/` is legacy and not read.

## Credit & License

This work is a re-implementation of the mesh-fitting stage of
**[4DSegment3.0](https://github.com/ImperialCollegeLondon/4DSegment3.0)** (Imperial
College London), derived from
[4DSegment2.0](https://github.com/ImperialCollegeLondon/4DSegment2.0) and the original
[j-duan/4Dsegment](https://github.com/j-duan/4Dsegment), and built on the IRTK/MIRTK
cardiac tools. **If you use this, please cite the 4DSegment papers and repositories.**
The registration and feature logic are theirs; only the GPU/PyTorch implementation is
new here.

MIT (inherited from the 4DSegment / Imperial College London project, from which the
bundled template originates). The registration approach is reimplemented from
scratch in PyTorch; IRTK/MIRTK themselves are separately licensed open-source
projects and are **not** redistributed here.
