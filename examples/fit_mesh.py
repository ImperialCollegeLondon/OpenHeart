"""Minimal end-to-end example: fit one subject and compute features.

Usage:
    python examples/fit_one_subject.py SEG_ED.nii.gz [SEG_ES.nii.gz] OUT_DIR
"""

import sys

import numpy as np

from openheart.mesh import FFDConfig, features, fit_subject
from openheart.mesh.template import load_template_mesh


def main():
    seg_ed = sys.argv[1]
    seg_es = sys.argv[2] if len(sys.argv) > 3 else None
    out_dir = sys.argv[-1]

    cfg = FFDConfig(iterations=300, control_spacing=20.0)
    results = fit_subject(seg_ed, seg_es, out_dir=out_dir, ffd_config=cfg)

    for phase, res in results.items():
        print(f"\n== {phase} ==")
        for region, mesh in res.meshes.items():
            m = res.metrics[region]
            print(f"  {region:7s} {mesh.n_points:6d} pts  fit={m['mean_surf_dist']:.2f}mm")

        if "LVendo" in res.meshes and "LVepi" in res.meshes:
            wt = features.wall_thickness(res.meshes["LVendo"], res.meshes["LVepi"])
            print(f"  LV wall thickness: {wt.mean():.2f} +/- {wt.std():.2f} mm")


if __name__ == "__main__":
    main()
