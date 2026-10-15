"""Validate the artifacts of the last pipeline run.

    python validate.py [--input-dir .] [--out-dir .]

Exits 0 when every check passes, 1 otherwise.
"""

import argparse
import sys

from pipeline.checks import run_checks


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate inspection-triage pipeline artifacts")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    results = run_checks(args.input_dir, args.out_dir, require_finalised=True)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}")
        for problem in r.problems:
            print(f"      - {problem}")
    failed = sum(not r.passed for r in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
