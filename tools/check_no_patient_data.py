#!/usr/bin/env python3
"""
Block patient data from entering the repository.

Used by the pre-commit hook (.githooks/pre-commit, checks staged files) and by
CI (--all, checks every tracked file). Exits 1 and lists offending files if:

  * the file type is a medical image / mesh / tabular / log / archive format
    (patient data or derived from it), or an image outside docs/;
  * the file is larger than MAX_BYTES;
  * a text file contains a DICOM UID or a hardcoded personal storage path.

A single line can be exempted by putting "patient-data-check: ignore" on it.
"""

import argparse
import re
import subprocess
import sys

MAX_BYTES = 2 * 1024 * 1024

BLOCKED_SUFFIXES = (
    # medical images / registration outputs
    ".nii", ".nii.gz", ".dcm", ".dicom", ".mha", ".mhd", ".nrrd", ".dof", ".dof.gz",
    # meshes
    ".vtk", ".vtp", ".vtu", ".stl", ".ply",
    # tabular / per-case results / logs
    ".json", ".csv", ".tsv", ".xls", ".xlsx", ".log",
    # arrays / archives
    ".npy", ".npz", ".h5", ".hdf5", ".mat", ".pkl", ".zip", ".tar", ".tar.gz", ".tgz", ".gz",
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp")
IMAGE_ALLOWED_PREFIXES = ("docs/",)
# Non-patient files allowed despite a blocked suffix. Only the public 4DSegment3.0
# atlas bundled with openheart.mesh belongs here — never add subject data.
ALLOWED_PATHS = set()
ALLOWED_PREFIXES = ("openheart/mesh/assets/template/",)

CONTENT_PATTERNS = [
    ("DICOM UID", re.compile(rb"\b1\.2\.840\.\d+(?:\.\d+){3,}|\b1\.3\.6\.1\.4\.1\.\d+(?:\.\d+){3,}")),
    ("personal storage path", re.compile(rb"/mnt/storage/home/|/home/ym\d+")),
]
IGNORE_MARKER = b"patient-data-check: ignore"
SELF = "tools/check_no_patient_data.py"


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def list_files(check_all):
    if check_all:
        out = git("ls-files", "-z")
    else:
        out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.decode().split("\0") if p]


def read_blob(path, check_all):
    # Staged mode reads the index so what is checked is exactly what gets committed.
    return git("show", f":{path}") if not check_all else open(path, "rb").read()


def check(path, data):
    problems = []
    lower = path.lower()
    allowed = path in ALLOWED_PATHS or path.startswith(ALLOWED_PREFIXES)
    if not allowed and lower.endswith(BLOCKED_SUFFIXES):
        problems.append("blocked file type (possible patient data)")
    if lower.endswith(IMAGE_SUFFIXES) and not path.startswith(IMAGE_ALLOWED_PREFIXES):
        problems.append("image outside docs/ (could show patient scans)")
    if len(data) > MAX_BYTES:
        problems.append(f"file is {len(data) / 1e6:.1f} MB (limit {MAX_BYTES / 1e6:.0f} MB)")
    if path != SELF and b"\0" not in data[:8000]:
        for lineno, line in enumerate(data.splitlines(), 1):
            if IGNORE_MARKER in line:
                continue
            for label, pattern in CONTENT_PATTERNS:
                if pattern.search(line):
                    problems.append(f"line {lineno}: {label}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="check all tracked files instead of staged ones")
    args = parser.parse_args()

    failures = {}
    for path in list_files(args.all):
        problems = check(path, read_blob(path, args.all))
        if problems:
            failures[path] = problems

    if failures:
        print("Patient-data check FAILED:", file=sys.stderr)
        for path, problems in failures.items():
            for p in problems:
                print(f"  {path}: {p}", file=sys.stderr)
        print("\nRemove these files/lines. Do not bypass with --no-verify without checking "
              "that no patient data is involved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
