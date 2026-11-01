"""Segment, post-process, fit and measure every case under a folder.

    python examples/run_cohort.py CASES_DIR OUT_DIR MODEL_DIR

CASES_DIR holds one folder per case with lvsa_ED.nii.gz / lvsa_ES.nii.gz (the
"legacy" layout); MODEL_DIR is the nnU-Net results folder
(…/Dataset003_RBHSHare/nnUNetTrainer__nnUNetPlans__2d). Re-running resumes.
"""

import logging
import sys

import openheart as oh


def main(cases_dir, out_dir, model_dir):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pipe = oh.Pipeline([
        oh.Segment(model_dir, folds=[1]),
        oh.Realign(),
        oh.FitMesh(),
        oh.Measure(),
    ])

    # One case first: everything stays in memory, nothing is written.
    cohort = oh.Cohort.load(cases_dir, layout="legacy")
    case = pipe(cohort[0])
    print(case)
    print({k: round(v, 1) for k, v in case.metrics.items() if k.startswith("lv_")})

    # Then the whole cohort, checkpointed under out_dir.
    results = pipe.run(cohort, out_dir, jobs=2)
    print(results)
    for failed in results.failed:
        print("failed:", failed.id, failed.error)
    df = results.to_dataframe()
    print(df[["case_id", "lv_edv_ml", "lv_ef_pct", "lv_mass_g", "wt_mean_ed_mm"]].describe())


if __name__ == "__main__":
    main(*sys.argv[1:4])
