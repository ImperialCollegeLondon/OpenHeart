#!/usr/bin/env python3
"""Generate docs/assets/pipeline.svg — the animated pipeline banner in the README.

Renders one case processed by OpenHeart through its four stages (cine MRI →
segmentation → fitted template mesh → wall-thickness map) and packs the frames into
a single SVG that animates with SMIL/CSS only (no scripts, no web fonts), so GitHub
plays it inside <img>. The README banner uses ACDC patient072 (public dataset) with the
segmentation from the cardiac QC agent (slice alignment + apex completion, gpt-5.5 run of
the AQC dataset); OpenHeart fits the meshes and measures wall thickness:

    openheart run agent_fixed_cases/ -o out/ --layout legacy --labels acdc
    python docs/assets/make_pipeline_svg.py out/patient072

Needs the [mesh] extra plus matplotlib and pillow.
"""

import argparse
import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from openheart import Case  # noqa: E402
from openheart.labels import CANONICAL  # noqa: E402

W, H = 940, 262
CARD_W, CARD_H, CARD_Y = 190, 212, 24
XS = [20, 254, 488, 722]
CYCLE = 6.0      # seconds for the card highlight to pass through all four stages
BEAT = 1.2       # seconds per ED→ES→ED heartbeat
SPIN = 6.0       # seconds per mesh revolution
N_FRAMES = 72    # mesh frames per revolution (5° apart)
TILE = 124       # displayed size of each panel image (px)
SCALE = 2        # render at 2x for sharp display on retina screens
WT_RANGE = (2.0, 12.0)  # mm, colour scale of the wall-thickness map

TEAL, LV, MYO, RV = "#2a9d8f", "#e76f51", "#2a9d8f", "#4c8bf5"
MUTED = "#8b949e"

STAGES = [
    ("Cine MRI", "short-axis, ED ↔ ES"),
    ("Segment", "nnU-Net"),
    ("Fit mesh", "template correspondence"),
    ("Measure", "wall thickness"),
]


# ----------------------------------------------------------------------------- images

def png_data_uri(img, colors=None):
    """PIL image → base64 PNG data URI, palette-quantised to keep the SVG small."""
    if colors:
        img = img.quantize(colors=colors, method=PILImage.FASTOCTREE, dither=PILImage.NONE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mri_tiles(case):
    """Mid-ventricular ED/ES crops (PIL): greyscale image and colour label overlay."""
    seg_ed = case.segs["ED"].array
    lv_slices = sorted({int(k) for k in np.argwhere(seg_ed == CANONICAL.lv)[:, 2]})
    k = lv_slices[len(lv_slices) // 2]
    heart = np.argwhere(seg_ed[..., k] > 0)
    ci, cj = heart.mean(0).round().astype(int)
    half = int(np.ceil(np.abs(heart - (ci, cj)).max() * 1.45))
    size = TILE * SCALE
    tiles = {}
    for phase in ("ED", "ES"):
        img = np.pad(case.images[phase].array[..., k], half)[ci:ci + 2 * half, cj:cj + 2 * half].T
        seg = np.pad(case.segs[phase].array[..., k], half)[ci:ci + 2 * half, cj:cj + 2 * half].T
        lo, hi = np.percentile(img[img > 0], [1, 99.5])
        grey = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        tiles[f"img_{phase}"] = PILImage.fromarray(grey).resize((size, size), PILImage.BICUBIC)
        rgba = np.zeros(seg.shape + (4,), np.uint8)
        for label, color in ((CANONICAL.rv, RV), (CANONICAL.myo, MYO), (CANONICAL.lv, LV)):
            rgba[seg == label] = hex_rgb(color) + (170,)
        tiles[f"seg_{phase}"] = PILImage.fromarray(rgba, "RGBA").resize((size, size), PILImage.NEAREST)
    return tiles


# ----------------------------------------------------------------------------- meshes

def decimated(mesh, scalars=None, reduction=0.6):
    """Decimate a Mesh with VTK; returns verts, faces and the scalars mapped by nearest vertex."""
    import vtk
    from scipy.spatial import cKDTree
    from vtk.util.numpy_support import vtk_to_numpy

    dec = vtk.vtkDecimatePro()
    dec.SetInputData(mesh.to_polydata())
    dec.SetTargetReduction(reduction)
    dec.PreserveTopologyOn()
    dec.Update()
    out = dec.GetOutput()
    verts = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    if scalars is not None:
        scalars = np.asarray(scalars)[cKDTree(mesh.verts).query(verts)[1]]
    return verts, faces, scalars


def long_axis_frame(myo_verts, rv_verts, slice_axis):
    """Rotation putting the LV long axis on z (base up) and the RV towards -x.

    Short-axis slices are cut perpendicular to the LV long axis, so the stack
    direction (the image affine's k column) is the long axis; the base is the end
    where the myocardium is wider.
    """
    axis = slice_axis / np.linalg.norm(slice_axis)
    centre = myo_verts.mean(0)
    t = (myo_verts - centre) @ axis
    radial = np.linalg.norm((myo_verts - centre) - np.outer(t, axis), axis=1)
    if radial[t > np.percentile(t, 80)].mean() < radial[t < np.percentile(t, 20)].mean():
        axis = -axis
    to_rv = rv_verts.mean(0) - centre
    x = -(to_rv - (to_rv @ axis) * axis)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(axis, x), axis])


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def render_turntable(parts, centre, rot, extent, elev=18):
    """parts: [(verts, faces, per-vertex RGB)]. Smooth (per-vertex) Lambert shading, light
    fixed to the camera. Returns N_FRAMES RGBA PIL images of one revolution."""
    frames = []
    size_in = TILE * SCALE / 100
    view = np.radians([elev, -60])
    to_camera = np.array([np.cos(view[0]) * np.cos(view[1]), np.cos(view[0]) * np.sin(view[1]),
                          np.sin(view[0])])
    light = to_camera + np.array([0.0, 0.0, 0.35])
    light /= np.linalg.norm(light)
    for f in range(N_FRAMES):
        a = 2 * np.pi * f / N_FRAMES
        spin = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        tris, colors = [], []
        for verts, faces, rgb in parts:
            v = (verts - centre) @ rot.T @ spin.T
            shade = 0.38 + 0.62 * np.abs(vertex_normals(v, faces) @ light)  # two-sided
            tris.append(v[faces])
            colors.append(np.clip((rgb * shade[:, None])[faces].mean(1), 0, 1))
        fig = plt.figure(figsize=(size_in, size_in), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(colors),
                                             edgecolors="none", linewidths=0, antialiased=False))
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=-60)
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        frames.append(PILImage.open(buf).convert("RGBA"))
    return frames


def mesh_frames(case):
    """Turntables of the fitted LV myocardium + RV, and of the wall-thickness map (PIL)."""
    meshes = case.meshes["ED"]
    thickness = case.features["ED"]["LVmyo"]["WallThickness"]
    myo_v, myo_f, wt = decimated(meshes["LVmyo"], thickness)
    rv_v, rv_f, _ = decimated(meshes["RV"])
    rot = long_axis_frame(meshes["LVmyo"].verts, meshes["RV"].verts, case.segs["ED"].affine[:3, 2])
    centre = np.concatenate([myo_v, rv_v]).mean(0)
    extent = np.abs((np.concatenate([myo_v, rv_v]) - centre) @ rot.T).max() * 0.66
    fit = render_turntable(
        [(myo_v, myo_f, np.tile(np.array(hex_rgb(MYO)) / 255, (len(myo_v), 1))),
         (rv_v, rv_f, np.tile(np.array(hex_rgb(RV)) / 255, (len(rv_v), 1)))],
        centre, rot, extent)
    wt_rgb = cm.turbo(np.clip((wt - WT_RANGE[0]) / (WT_RANGE[1] - WT_RANGE[0]), 0, 1))[:, :3]
    myo_centre = myo_v.mean(0)
    thick = render_turntable([(myo_v, myo_f, wt_rgb)], myo_centre, rot,
                             np.abs((myo_v - myo_centre) @ rot.T).max() * 0.68)
    return fit, thick


# ----------------------------------------------------------------------------- svg

def flipbook(frames, x, y, size, dur):
    n = len(frames)
    key_times = ";".join(f"{i / n:.5f}" for i in range(n))
    parts = []
    for i, uri in enumerate(frames):
        values = ";".join("1" if j == i else "0" for j in range(n))
        parts.append(f'<image href="{uri}" x="{x}" y="{y}" width="{size}" height="{size}" opacity="{1 if i == 0 else 0}">'
                     f'<animate attributeName="opacity" values="{values}" keyTimes="{key_times}" dur="{dur}s" '
                     f'calcMode="discrete" repeatCount="indefinite"/></image>')
    return "".join(parts)


def beat(uri_ed, uri_es, x, y, size):
    """ED image with the ES frame flashing on top: a two-frame heartbeat."""
    return (f'<image href="{uri_ed}" x="{x}" y="{y}" width="{size}" height="{size}"/>'
            f'<image href="{uri_es}" x="{x}" y="{y}" width="{size}" height="{size}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1" keyTimes="0;0.55" calcMode="discrete" '
            f'dur="{BEAT}s" repeatCount="indefinite"/></image>')


def tile(cx, cy, clip_id, content):
    x, y = cx - TILE / 2, cy - TILE / 2
    return (f'<clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{TILE}" height="{TILE}" rx="10"/></clipPath>'
            f'<g clip-path="url(#{clip_id})">{content(x, y)}</g>')


def stage_mri(cx, cy, m):
    return tile(cx, cy, "tile0", lambda x, y: beat(m["img_ED"], m["img_ES"], x, y, TILE))


def stage_segment(cx, cy, m):
    return tile(cx, cy, "tile1", lambda x, y: beat(m["img_ED"], m["img_ES"], x, y, TILE)
                + beat(m["seg_ED"], m["seg_ES"], x, y, TILE))


def stage_mesh(cx, cy, frames):
    return flipbook(frames, cx - TILE / 2, cy - TILE / 2, TILE, SPIN)


def stage_thickness(cx, cy, frames):
    bar_w, bar_y = 104, cy + TILE / 2 + 4
    stops = "".join(f'<stop offset="{t:.2f}" stop-color="{matplotlib.colors.to_hex(cm.turbo(t))}"/>'
                    for t in np.linspace(0, 1, 9))
    return (flipbook(frames, cx - TILE / 2, cy - TILE / 2 - 6, TILE, SPIN)
            + f'<linearGradient id="wt">{stops}</linearGradient>'
            f'<rect x="{cx - bar_w / 2}" y="{bar_y}" width="{bar_w}" height="5" rx="2.5" fill="url(#wt)"/>'
            f'<text x="{cx - bar_w / 2}" y="{bar_y + 15}" class="small">{WT_RANGE[0]:g}</text>'
            f'<text x="{cx + bar_w / 2}" y="{bar_y + 15}" class="small" text-anchor="end">{WT_RANGE[1]:g} mm</text>')


def connector(i):
    x0, x1, y = XS[i] + CARD_W + 6, XS[i + 1] - 6, CARD_Y + 86
    t0 = (i + 1) * 0.25 - 0.05  # dot travels just before the next stage lights up
    kt = f"0;{t0:.3f};{t0 + 0.05:.3f};1"
    return (f'<line x1="{x0}" x2="{x1 - 6}" y1="{y}" y2="{y}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="3 4"/>'
            f'<path d="M{x1 - 7} {y - 4.5} L{x1} {y} L{x1 - 7} {y + 4.5}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>'
            f'<circle cx="{x0}" cy="{y}" r="4" fill="{TEAL}">'
            f'<animate attributeName="cx" values="{x0};{x0};{x1};{x1}" keyTimes="{kt}" dur="{CYCLE}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;1;0;0" keyTimes="{kt}" dur="{CYCLE}s" repeatCount="indefinite"/>'
            f'</circle>')


def build(mri, fit, thick):
    font = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
    style = f"""
    .title {{ font: 600 16px {font}; fill: {TEAL}; }}
    .sub   {{ font: 12px {font}; fill: {MUTED}; }}
    .small {{ font: 10px {font}; fill: {MUTED}; }}
    .card  {{ fill: rgba(127,127,127,0.06); stroke: rgba(127,127,127,0.35); stroke-width: 1;
             animation: glow {CYCLE}s infinite; }}
    @keyframes glow {{
      0%, 22% {{ stroke: {TEAL}; stroke-width: 2.5; }}
      27%, 100% {{ stroke: rgba(127,127,127,0.35); stroke-width: 1; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ .card {{ animation: none; }} }}
    """
    drawers = [lambda cx, cy: stage_mri(cx, cy, mri), lambda cx, cy: stage_segment(cx, cy, mri),
               lambda cx, cy: stage_mesh(cx, cy, fit), lambda cx, cy: stage_thickness(cx, cy, thick)]
    body = []
    for i, (x, (title, sub), draw) in enumerate(zip(XS, STAGES, drawers)):
        cx, cy = x + CARD_W / 2, CARD_Y + 86
        body.append(f'<rect class="card" x="{x}" y="{CARD_Y}" width="{CARD_W}" height="{CARD_H}" rx="14" '
                    f'style="animation-delay:{i * CYCLE / 4}s"/>')
        body.append(draw(cx, cy))
        body.append(f'<text x="{cx}" y="{CARD_Y + 184}" class="title" text-anchor="middle">{title}</text>')
        body.append(f'<text x="{cx}" y="{CARD_Y + 201}" class="sub" text-anchor="middle">{sub}</text>')
    body.extend(connector(i) for i in range(3))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="OpenHeart pipeline on a real case: cine MRI, segmentation, mesh fitting, wall thickness">'
            f'<style>{style}</style>{"".join(body)}</svg>\n')


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("case", help="a case folder written by `openheart run` (openheart layout)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.svg"))
    args = p.parse_args()

    case = Case.load(args.case)
    mri = {k: png_data_uri(v, colors=None if k.startswith("seg") else 64) for k, v in mri_tiles(case).items()}
    fit, thick = mesh_frames(case)
    svg = build(mri, [png_data_uri(f, colors=64) for f in fit], [png_data_uri(f, colors=96) for f in thick])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {args.out} ({len(svg) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
