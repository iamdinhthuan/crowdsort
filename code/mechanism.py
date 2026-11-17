"""Speaker-level error analysis by arrival rank (CPU only).

For every recording, hypothesis channels (thresholded at 0.5) are mapped one-to-one to reference speakers by the
optimal assignment on overlapping speech (10 ms grid). For each reference speaker this gives
- its arrival rank (order of first speech in the reference),
- whether it is found: mapped to a channel and at least 50% of its speech is covered by that channel,
- whether it is merged: unmapped or not found, but at least 50% of its speech is covered by channels that are
  mapped to other speakers,
- its first-detection delay: time from its first reference speech to the first frame of its speech covered by
  its own channel (speakers never covered are excluded),
- the speaker-attributed error rate of its speech before and after a return (speech that resumes after at least
  10 s without this speaker), i.e. the fraction of its reference speech not covered by its own channel.

python mechanism.py --sets msdwild_many,avaavd_test --systems N3D=base FTclips=c3_noconcat+c3_noconcat_s2+...
Writes results/mechanism.json and prints a table per set.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = "/mnt/ssd2/paper/paper_1"
RET_GAP = 10.0
BINS = [(1, 2), (3, 4), (5, 6), (7, 99)]


def read_rttm(path):
    out = []
    for line in open(path):
        p = line.split()
        if p and p[0] == "SPEAKER":
            out.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(out)


def analyse(item, pred_dir):
    uid = os.path.basename(item["audio_filepath"])[:-4]
    d = np.load(os.path.join(pred_dir, uid + ".npz"))
    hyp = d["probs"].astype(np.float32) > 0.5
    T = hyp.shape[0]
    segs = read_rttm(item["rttm_filepath"])
    spks = []
    for s, e, k in segs:
        if k not in spks:
            spks.append(k)
    R = np.zeros((T, len(spks)), bool)
    after_return = np.zeros((T, len(spks)), bool)
    last_end = {}
    for s, e, k in segs:
        j = spks.index(k)
        a, b = min(int(round(s * 100)), T), min(int(round(e * 100)), T)
        R[a:b, j] = True
        if k in last_end and s - last_end[k] >= RET_GAP:
            after_return[a:, j] = True
        last_end[k] = max(last_end.get(k, 0.0), e)
    after_return &= R
    ov = R.T.astype(np.float32) @ hyp.astype(np.float32)  # speakers x channels
    ri, ci = linear_sum_assignment(-ov)
    chan = {r: c for r, c in zip(ri, ci) if ov[r, c] > 0}
    rows = []
    for j, k in enumerate(spks):
        sp = R[:, j]
        n = sp.sum()
        if n == 0:
            continue
        own = hyp[:, chan[j]] if j in chan else np.zeros(T, bool)
        others = np.zeros(T, bool)
        for jj, c in chan.items():
            if jj != j:
                others |= hyp[:, c]
        cov = (own & sp).sum() / n
        found = cov >= 0.5
        merged = (not found) and ((others & sp & ~own).sum() / n >= 0.5)
        first = np.argmax(sp)
        hit = np.where(own & sp)[0]
        delay = (hit[0] - first) / 100 if len(hit) else None
        pre, post = sp & ~after_return[:, j], after_return[:, j]
        rows.append({"rank": j + 1, "found": bool(found), "merged": bool(merged), "delay": delay,
                     "pre_n": int(pre.sum()), "pre_err": int((pre & ~own).sum()),
                     "post_n": int(post.sum()), "post_err": int((post & ~own).sum())})
    return rows


def summarise(rows):
    out = {}
    for lo, hi in BINS:
        r = [x for x in rows if lo <= x["rank"] <= hi]
        if not r:
            continue
        dl = [x["delay"] for x in r if x["delay"] is not None]
        out[f"{lo}-{hi if hi < 99 else '+'}"] = {
            "n": len(r), "found_pct": round(100 * np.mean([x["found"] for x in r]), 1),
            "merged_pct": round(100 * np.mean([x["merged"] for x in r]), 1),
            "delay_median_s": round(float(np.median(dl)), 2) if dl else None}
    pre_n, post_n = sum(x["pre_n"] for x in rows), sum(x["post_n"] for x in rows)
    out["err_before_return_pct"] = round(100 * sum(x["pre_err"] for x in rows) / max(1, pre_n), 1)
    out["err_after_return_pct"] = round(100 * sum(x["post_err"] for x in rows) / max(1, post_n), 1)
    out["after_return_share_pct"] = round(100 * post_n / max(1, pre_n + post_n), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="NAME=model1+model2 (seeds pooled)")
    ap.add_argument("--profile", default="1.04")
    a = ap.parse_args()
    res = {}
    for s in a.sets.split(","):
        items = [json.loads(l) for l in open(f"{ROOT}/data/manifests/{s}.jsonl")]
        res[s] = {}
        for spec in a.systems:
            name, models = spec.split("=")
            rows = []
            for m in models.split("+"):
                pd = f"{ROOT}/exp/{m}/{s}/{a.profile}"
                if not os.path.isdir(pd):
                    continue
                for it in items:
                    rows += analyse(it, pd)
            if rows:
                res[s][name] = summarise(rows)
    path = f"{ROOT}/results/mechanism.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    for s in res:
        old.setdefault(s, {}).update(res[s])
    json.dump(old, open(path, "w"), indent=1)
    for s, d in res.items():
        print(f"== {s}")
        for name, v in d.items():
            bins = "  ".join(f"r{k}: found {b['found_pct']}% merged {b['merged_pct']}% delay {b['delay_median_s']}s (n={b['n']})"
                             for k, b in v.items() if isinstance(b, dict))
            print(f"{name:12s} {bins}\n{'':12s} err before/after return {v['err_before_return_pct']}/{v['err_after_return_pct']}% "
                  f"(after-return share {v['after_return_share_pct']}%)")


if __name__ == "__main__":
    main()
