"""``openheart`` command line.

    openheart run INPUT -o OUT [--model MODEL] [--realign] [--fill-apex] [--jobs N]
    openheart fit --ed seg_ED.nii.gz [--es seg_ES.nii.gz] --out out/ [--features]
    openheart export qc OUT -o qc.json
    openheart export nnunet INPUT... -o NNUNET_RAW --dataset-id 3 --name RBHSHare
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .layouts import LAYOUTS, Layout


def _input_layout(args):
    """The named ``--layout``, or a layout built from ``--image`` / ``--seg`` patterns."""
    if not (args.image or args.seg):
        return args.layout
    return Layout(name="custom", image=(args.image,) if args.image else (), seg=args.seg or "",
                  frames=args.frames, writable=False)


def _run(args) -> int:
    from .case import Cohort
    from .fit import FitMesh
    from .measure import Measure
    from .pipeline import Pipeline
    from .postprocess import FillApex, Realign

    steps = []
    if args.model:
        from .segment import Segment

        steps.append(Segment(args.model, folds=args.folds, checkpoint=args.checkpoint, device=args.device))
    if args.realign:
        steps.append(Realign())
    if args.fill_apex:
        steps.append(FillApex())
    if not args.no_mesh:
        steps.append(FitMesh(device=args.device))
    steps.append(Measure(contour_thickness=args.contour_thickness))

    cohort = Cohort.load(args.input, layout=_input_layout(args), labels=args.labels)
    if not len(cohort):
        logging.error("no %s-layout cases found under %s", cohort.layout.name, args.input)
        return 1
    pipeline = Pipeline(steps)
    logging.info("%r", pipeline)
    results = pipeline.run(cohort, args.out, layout=args.out_layout, jobs=args.jobs,
                           resume=not args.no_resume, scheduler=args.scheduler)
    logging.info("metrics table: %s/metrics.csv", args.out)
    return 0 if not results.failed else 3


def _export(args) -> int:
    if args.target == "qc":
        from .export import qc_json

        n = qc_json(args.input[0], args.out)
        logging.info("wrote %d cases to %s", n, args.out)
        return 0
    from .case import Cohort
    from .training import export_nnunet_dataset, pairs_from_csv

    items = []
    for source in args.input:
        if source.endswith(".csv"):
            items += pairs_from_csv(source, args.base_dir)
        else:
            items += list(Cohort.load(source, layout=_input_layout(args), labels=args.labels))
    labels = args.labels if all(s.endswith(".csv") for s in args.input) else None
    export_nnunet_dataset(items, args.out, args.dataset_id, args.name, labels=labels)
    return 0


def _add_layout_arguments(parser, default: str) -> None:
    parser.add_argument("--layout", default=default, choices=sorted(LAYOUTS),
                        help=f"input layout (default: {default})")
    parser.add_argument("--labels", help="input label convention if not the layout's (e.g. 'lv=3,myo=2,rv=1')")
    custom = parser.add_argument_group("custom input layout (replaces --layout)")
    custom.add_argument("--image", help="image filename pattern, e.g. '{id}_SA_{phase}.nii.gz'")
    custom.add_argument("--seg", help="segmentation filename pattern, e.g. '{id}_SA_{phase}_gt.nii.gz'")
    custom.add_argument("--frames", help="per-case text file with 'ED: <n>' and 'ES: <n>' lines; "
                                         "patterns can then use {frame}, e.g. '{id}_frame{frame:02d}.nii.gz'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openheart", description="Cardiac MRI analysis SDK.")
    parser.add_argument("--version", action="version", version=f"openheart {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run = sub.add_parser("run", help="Run the pipeline over every case under a folder")
    run.add_argument("input", help="folder of cases (or a single case folder)")
    run.add_argument("-o", "--out", required=True, help="output folder (also the checkpoint for resuming)")
    _add_layout_arguments(run, default="legacy")
    run.add_argument("--out-layout", default="openheart", choices=["openheart", "legacy"],
                     help="output layout; 'legacy' is what the QC app reads (default: openheart)")
    seg = run.add_argument_group("segmentation (runs when --model is given)")
    seg.add_argument("--model", help="nnU-Net results folder (…/nnUNetTrainer__nnUNetPlans__2d) or model name")
    seg.add_argument("--folds", nargs="+", help="folds to ensemble (default: all found)")
    seg.add_argument("--checkpoint", default="checkpoint_final.pth")
    post = run.add_argument_group("post-processing and measurement")
    post.add_argument("--realign", action="store_true", help="straighten slice stacks along a fitted centreline")
    post.add_argument("--fill-apex", action="store_true", help="close an open LV apex")
    post.add_argument("--no-mesh", action="store_true", help="skip template mesh fitting (and 3D wall thickness)")
    post.add_argument("--contour-thickness", action="store_true", help="add 2D wall thickness with AHA segments")
    ex = run.add_argument_group("execution")
    ex.add_argument("--device", default="cuda", help="cuda or cpu")
    ex.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ex.add_argument("--no-resume", action="store_true", help="recompute cases already in OUT")
    ex.add_argument("--scheduler", default="local", choices=["local", "luigi", "luigi-central"])
    run.set_defaults(func=_run)

    # `fit` forwards its arguments to the mesh-fitting CLI, so its --help is the full one.
    sub.add_parser("fit", help="Fit the template mesh to one subject's segmentations", add_help=False)

    export = sub.add_parser("export", help="Export results for other tools")
    export.add_argument("target", choices=["qc", "nnunet"],
                        help="qc: review list for the QC web app; nnunet: a training dataset")
    export.add_argument("input", nargs="+", help="qc: an output folder in the legacy layout; "
                                                 "nnunet: case folders and/or image_path,label_path CSVs")
    export.add_argument("-o", "--out", required=True, help="qc: JSON file; nnunet: the nnUNet_raw folder")
    _add_layout_arguments(export, default="openheart")
    export.add_argument("--base-dir", default="", help="prefix for relative paths in CSVs")
    export.add_argument("--dataset-id", type=int, default=3)
    export.add_argument("--name", default="OpenHeart")
    export.set_defaults(func=_export)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "fit":
        from .mesh.cli import main as fit_main

        return fit_main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
