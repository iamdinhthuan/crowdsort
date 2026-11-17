"""Paired bootstrap over recordings for the pooled-DER difference between two systems.

python bootstrap.py A_perfile.json B_perfile.json [--n 10000]
Each per-file json maps uid -> {total, miss, fa, conf}. Prints DER_A, DER_B, delta (B - A),
95% CI of delta and the one-sided p-value P(delta >= 0) (i.e. B not better than A).
Several (A, B) pairs over different sets can be pooled by passing comma-separated lists.
--group resamples recordings by their identifier instead of by (set, identifier), so that the views of one
meeting in several microphone conditions (AMI MHM/SDM, AliMeeting near/far) are drawn together, and the three
5-min AVA-AVD clips of one film (<film>_c_01..03) form one group, as do the single-channel devices of one
NOTSOFAR-1 meeting (<meeting>_sc_<device>).
"""
import argparse
import json
import re

import numpy as np


def load(paths):
    """Comma separates test sets (pooled); '+' separates seeds of one set (per-file errors averaged)."""
    d = {}
    for i, group in enumerate(paths.split(",")):
        seeds = [json.load(open(p)) for p in group.split("+")]
        for k in seeds[0]:
            d[f"{i}:{k}"] = {m: sum(sd[k][m] for sd in seeds) / len(seeds) for m in ("total", "miss", "fa", "conf")}
    return d


def pooled(rows, idx):
    err = rows[idx, 1:].sum()
    return 100 * err / rows[idx, 0].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", action="store_true")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    ra = np.array([[A[k]["total"], A[k]["miss"], A[k]["fa"], A[k]["conf"]] for k in keys])
    rb = np.array([[B[k]["total"], B[k]["miss"], B[k]["fa"], B[k]["conf"]] for k in keys])
    rng = np.random.default_rng(args.seed)
    n = len(keys)
    all_idx = np.arange(n)
    da, db = pooled(ra, all_idx), pooled(rb, all_idx)
    gid = [re.sub(r"(_c_\d+|_sc_[a-z]+_\d+)$", "", k.split(":", 1)[1]) if args.group else k for k in keys]
    groups = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(i)
    members = list(groups.values())
    G = len(members)
    deltas = np.empty(args.n)
    for i in range(args.n):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([members[j] for j in pick])
        deltas[i] = pooled(rb, idx) - pooled(ra, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out = {"n_files": n, "n_groups": G, "DER_A": round(da, 2), "DER_B": round(db, 2), "delta": round(db - da, 2),
           "ci95": [round(lo, 2), round(hi, 2)], "p_B_not_better": round(float((deltas >= 0).mean()), 4),
           "rel_change_pct": round(100 * (db - da) / da, 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
