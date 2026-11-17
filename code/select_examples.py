"""Rule-based choice of visualization examples (no hand picking).

Rule: among MSDWild-many recordings with >= 7 reference speakers, take (a) the one whose DER change
(proposed - baseline) is the median, and (b) the one with the largest DER increase (failure case).
For AISHELL-4 test, take the session with the median late-session change and show minutes 30-32.
python select_examples.py --base ft3_s1 --prop sam3_s1
"""
import argparse
import json

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"


def der(r):
    return 100 * (r["miss"] + r["fa"] + r["conf"]) / r["total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prop", required=True)
    a = ap.parse_args()
    out = {}
    A = json.load(open(f"{ROOT}/exp/{a.base}/msdwild_many/1.04/_perfile_thr.json"))
    B = json.load(open(f"{ROOT}/exp/{a.prop}/msdwild_many/1.04/_perfile_thr.json"))
    cand = sorted([(der(B[k]) - der(A[k]), k) for k in A if A[k]["n_ref"] >= 7])
    out["msdwild_median"] = cand[len(cand) // 2]
    out["msdwild_failure"] = cand[-1]
    A = json.load(open(f"{ROOT}/exp/{a.base}/aishell4_test/1.04/_perfile_thr.json"))
    B = json.load(open(f"{ROOT}/exp/{a.prop}/aishell4_test/1.04/_perfile_thr.json"))
    c2 = sorted([(der(B[k]) - der(A[k]), k) for k in A])
    out["aishell4_median"] = c2[len(c2) // 2]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
