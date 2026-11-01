"""Command-line interface: fit a subject's segmentation(s) to the template.

Example
-------
    openheart fit --ed seg_lvsa_SR_ED.nii.gz --es seg_lvsa_SR_ES.nii.gz \
                  --out out/ --features
"""

from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="openheart fit", description="GPU cardiac template mesh fitting.")
    p.add_argument("--ed", required=True, help="ED segmentation NIfTI.")
    p.add_argument("--es", default=None, help="ES segmentation NIfTI (optional).")
    p.add_argument("--out", required=True, help="Output directory for fitted meshes.")
    p.add_argument(
        "--labels",
        default="4dseg",
        help="Segmentation label convention: an explicit 'lv=3,myo=2,rv=1'. "
        "Default 4dseg (1=LV,2=myo,3=RV).",
    )
    p.add_argument("--iters", type=int, default=300, help="FFD optimisation iterations.")
    p.add_argument("--control-spacing", type=float, default=20.0, help="FFD control-point spacing (mm).")
    p.add_argument("--device", default="cuda", help="cuda or cpu.")
    p.add_argument("--template", default=None, help="Template folder (default: the bundled atlas).")
    p.add_argument("--features", action="store_true", help="Also compute wall thickness / curvature / signed distance.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import sys

    from .backends.ffd_torch import FFDConfig
    from .errors import InsufficientLandmarksError
    from .fitting import fit_subject

    cfg = FFDConfig(
        iterations=args.iters, control_spacing=args.control_spacing, device=args.device
    )
    try:
        results = fit_subject(
            args.ed, args.es, out_dir=args.out, labels=args.labels,
            ffd_config=cfg, template=args.template, verbose=not args.quiet,
        )
    except InsufficientLandmarksError as e:
        # Incomplete segmentation (empty chamber) — report cleanly and exit non-zero
        # so batch runners record a skip instead of a traceback.
        print(f"SKIPPED: {e}", file=sys.stderr)
        return 2

    if args.features:
        from . import features
        from .template import load_template_mesh
        from .io.mesh import Mesh

        for phase, res in results.items():
            feat_dir = os.path.join(args.out, "mesh_features")
            os.makedirs(feat_dir, exist_ok=True)
            if {"LVendo", "LVepi", "LVmyo"} <= res.meshes.keys():
                # Faithful to 4DSegment3.0 (utils/imaging.py): thickness scalar
                # mapped onto the LVmyo mesh -> W_LVmyo_<phase>.vtk, then the .txt
                # is `vtk2txt`-exported from that mesh (x y z WallThickness).
                myo_wt = features.wall_thickness_myocardium(
                    res.meshes["LVendo"], res.meshes["LVepi"], res.meshes["LVmyo"]
                )
                myo_wt.write(os.path.join(args.out, f"W_LVmyo_{phase}.vtk"))
                myo_wt.write_txt(
                    os.path.join(feat_dir, f"lv_myo{phase}_wallthickness.txt"),
                    scalar="WallThickness",
                )
            if "LVmyo" in res.meshes:
                myo = res.meshes["LVmyo"]
                sd = features.signed_distance_to_template(myo, load_template_mesh("LVmyo", phase, args.template))
                cur = features.mean_curvature(myo, smooth_iters=2)
                # Carry each scalar on a copy of the LVmyo mesh and vtk2txt-export
                # it (x y z scalar), matching the original .txt layout.
                myo_sd = Mesh(verts=myo.verts.copy(), faces=myo.faces.copy(),
                              point_scalars={"DistanceProfile": sd})
                myo_cur = Mesh(verts=myo.verts.copy(), faces=myo.faces.copy(),
                               point_scalars={"Curvature": cur})
                myo_sd.write_txt(
                    os.path.join(feat_dir, f"lv_myo{phase}_signeddistances.txt"),
                    scalar="DistanceProfile",
                )
                myo_cur.write_txt(
                    os.path.join(feat_dir, f"lv_myo{phase}_curvature.txt"),
                    scalar="Curvature",
                )
            if not args.quiet:
                print(f"  features written for {phase} -> {feat_dir}")

    if not args.quiet:
        print("Done.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
