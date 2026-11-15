"""Diarization timelines: reference vs systems on the same excerpt (visualization results).

python fig_timeline.py --out paper/tex/fig_timeline.pdf \
   --example msdwild_many:01234:0:90 --example aishell4_test:L_R003S01C02:600:690 \
   --systems base:N3D,ft_base_s1:N3D-FT,sfx_full_s1:Proposed --profile 1.04
Hypothesis channels are mapped to reference speakers with the optimal (Hungarian) mapping over the whole
recording, so colours are comparable; unmapped channels are drawn in grey.
"""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

ROOT = CROWDSORT_ROOT
STYLE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figure.mplstyle")
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7A5195", "#B5A300", "#8C564B", "#1B9E77"]


def ref_matrix(rttm, n):
    segs = []
    for l in open(rttm):
        p = l.split()
        segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    first = {}
    for s, e, k in segs:
        first[k] = min(first.get(k, 1e9), s)
    spk = sorted(first, key=first.get)
    m = np.zeros((n, len(spk)), dtype=bool)
    for s, e, k in segs:
        m[int(round(s * 100)) : int(round(e * 100)), spk.index(k)] = True
    return m


def mapping(ref, hyp):
    n = min(len(ref), len(hyp))
    ov = ref[:n].T.astype(np.float32) @ hyp[:n].astype(np.float32)  # (R, H)
    r, c = linear_sum_assignment(-ov)
    mp = {}
    for i, j in zip(r, c):
        if ov[i, j] > 0:
            mp[j] = i
    return mp


def draw(ax, mat, colors, t0, t1, label):
    n_rows = mat.shape[1]
    for k in range(n_rows):
        x = np.concatenate([[0], mat[:, k].astype(np.int8), [0]])
        d = np.diff(x)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ax.broken_barh([(t0 + s / 100, (e - s) / 100)], (n_rows - 1 - k + 0.1, 0.8), facecolors=colors[k],
                           edgecolor="none")
    ax.set_ylim(0, max(n_rows, 1))
    ax.set_yticks([])
    ax.set_xlim(t0, t1)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=4)
    for side in ("left",):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", action="append", required=True, help="set:uid:start:end")
    ap.add_argument("--systems", required=True, help="model:label,...")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width-mm", type=float, default=190)
    a = ap.parse_args()
    plt.style.use(STYLE)
    systems = [s.split(":") for s in a.systems.split(",")]
    n_rows_total = len(a.example) * (len(systems) + 1)
    fig, axes = plt.subplots(n_rows_total, 1, figsize=(a.width_mm / 25.4, 0.55 * n_rows_total + 0.4),
                             sharex=False)
    row = 0
    for ex in a.example:
        sset, uid, t0, t1 = ex.split(":")
        t0, t1 = float(t0), float(t1)
        dur = [json.loads(l)["duration"] for l in open(f"{ROOT}/data/manifests/{sset}.jsonl")
               if os.path.basename(json.loads(l)["audio_filepath"])[:-4] == uid][0]
        t1 = min(t1, dur)
        item = [json.loads(l) for l in open(f"{ROOT}/data/manifests/{sset}.jsonl")
                if os.path.basename(json.loads(l)["audio_filepath"])[:-4] == uid][0]
        probs = {m: np.load(f"{ROOT}/exp/{m}/{sset}/{a.profile}/{uid}.npz")["probs"].astype(np.float32)
                 for m, _ in systems}
        n = max(p.shape[0] for p in probs.values())
        ref = ref_matrix(item["rttm_filepath"], n)
        i0, i1 = int(t0 * 100), int(t1 * 100)
        draw(axes[row], ref[i0:i1], COLORS, t0, t1, "Reference")
        axes[row].set_title(f"{sset.replace('_', ' ')} / {uid}", loc="left", fontsize=7)
        row += 1
        for m, lab in systems:
            hyp = probs[m] > 0.5
            mp = mapping(ref, hyp)
            # rows follow the reference order; unmatched hypothesis channels appended in grey
            rows, cols = [], []
            for r_i in range(ref.shape[1]):
                js = [j for j, rr in mp.items() if rr == r_i]
                rows.append(hyp[:, js[0]] if js else np.zeros(hyp.shape[0], bool))
                cols.append(COLORS[r_i % len(COLORS)])
            for j in range(hyp.shape[1]):
                if j not in mp and hyp[i0:i1, j].any():
                    rows.append(hyp[:, j])
                    cols.append("#9E9E9E")
            mat = np.stack(rows, 1)[i0:i1]
            draw(axes[row], mat, cols, t0, t1, lab)
            row += 1
        axes[row - 1].set_xlabel("Time (s)")
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
