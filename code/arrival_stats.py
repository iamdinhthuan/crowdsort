"""Speaker arrival and return statistics of a manifest (CPU only).

For every recording: number of speakers, number of late arrivals (speakers whose first speech starts
after the cumulative speech has exceeded the time span of the default speaker cache, 264 frames x 80 ms)
and number of returns (a speaker resumes after at least `gap` seconds during which another speaker spoke).

python arrival_stats.py set1 set2 ...   (names of data/manifests/<set>.jsonl)
"""
import json
import sys

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"
CACHE_SPAN = 264 * 0.08


def rec_stats(rttm, gap=10.0):
    segs = []
    for line in open(rttm):
        p = line.split()
        if p and p[0] == "SPEAKER" and float(p[4]) > 0:
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    segs.sort()
    first, last_end, last_start = {}, {}, {}
    speech, late, returns = 0.0, 0, 0
    cursor = 0.0
    for s, e, k in segs:
        if k not in first:
            first[k] = s
            if speech >= CACHE_SPAN:
                late += 1
        else:
            le = last_end[k]
            others = any(t > le for k2, t in last_start.items() if k2 != k)
            if s - le >= gap and others:
                returns += 1
        last_end[k] = max(last_end.get(k, 0.0), e)
        last_start[k] = s
        speech += max(0.0, e - max(s, cursor))
        cursor = max(cursor, e)
    return len(first), late, returns, segs[-1][1] if segs else 0.0


def main():
    print(f"{'set':22s} {'n':>5s} {'spk':>5s} {'>4spk%':>7s} {'late/rec':>9s} {'late%spk':>9s} {'ret/min':>8s}")
    for name in sys.argv[1:]:
        rows = []
        for line in open(f"{ROOT}/data/manifests/{name}.jsonl"):
            d = json.loads(line)
            rows.append(rec_stats(d["rttm_filepath"]) + (d["duration"],))
        a = np.array(rows, dtype=float)
        nspk, late, ret, _, dur = a.T
        print(f"{name:22s} {len(a):5d} {nspk.mean():5.2f} {100 * (nspk > 4).mean():7.1f} {late.mean():9.2f} "
              f"{100 * late.sum() / nspk.sum():9.1f} {60 * ret.sum() / dur.sum():8.2f}")


if __name__ == "__main__":
    main()
