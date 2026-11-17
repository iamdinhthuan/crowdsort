"""Diarization timelines: reference vs systems on the same excerpt (visualization results).

python fig_timeline.py --out paper/tex/fig_timeline.pdf \
   --example msdwild_many:01234:0:90 --example aishell4_test:L_R003S01C02:600:690 \
   --systems base:N3D,ft_base_s1:N3D-FT,sfx_full_s1:Proposed --profile 1.04
Hypothesis channels are mapped to reference speakers with the optimal (Hungarian) mapping over the whole
recording, so colours are comparable; unmapped channels are drawn in grey.
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS  # noqa: E402

ROOT = "/mnt/ssd2/paper/paper_1"
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csl-figure.mplstyle")
COLORS = FS.SPEAKERS


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


def draw(ax, mat, colors, t0, t1, label, bold=False):
    n_rows = mat.shape[1]
    for k in range(n_rows):
        x = np.concatenate([[0], mat[:, k].astype(np.int8), [0]])
        d = np.diff(x)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ax.broken_barh([(t0 + s / 100, (e - s) / 100)], (n_rows - 1 - k + 0.12, 0.76), facecolors=colors[k],
                           edgecolor="none")
    ax.set_ylim(0, max(n_rows, 1))
    ax.set_yticks([])
    ax.set_xlim(t0, t1)
    ax.grid(False)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=3,
                  fontweight="bold" if bold else "normal")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(FS.GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", action="append", required=True, help="set:uid:start:end")
    ap.add_argument("--title", action="append", default=[], help="panel title, one per example")
    ap.add_argument("--systems", required=True, help="model:label,...")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bold", default="", help="label of the row to set in bold")
    a = ap.parse_args()
    FS.apply()
    systems = [s.split(":") for s in a.systems.split(",")]
    nrow = len(systems) + 1
    row_h = 0.18  # inches per row
    heights = []
    for _ in a.example:
        heights += [1] * nrow + [2.6]
    heights = heights[:-1]
    fig_h = row_h * sum(heights) + 0.25
    fig, axes = plt.subplots(len(heights), 1, figsize=(FS.TEXT_W, fig_h), gridspec_kw={"height_ratios": heights,
                             "hspace": 0.08, "left": 0.105, "right": 0.982, "top": 0.97, "bottom": 0.06})
    gap_axes = [axes[i] for i in range(len(heights)) if (i + 1) % (nrow + 1) == 0]
    for g in gap_axes:
        g.axis("off")
    axes = [ax for ax in axes if ax not in gap_axes]
    row = 0
    for ei, ex in enumerate(a.example):
        sset, uid, t0, t1 = ex.split(":")
        t0, t1 = float(t0), float(t1)
        items = [json.loads(l) for l in open(f"{ROOT}/data/manifests/{sset}.jsonl")]
        item = [it for it in items if os.path.basename(it["audio_filepath"])[:-4] == uid][0]
        t1 = min(t1, item["duration"])
        probs = {m: np.load(f"{ROOT}/exp/{m}/{sset}/{a.profile}/{uid}.npz")["probs"].astype(np.float32)
                 for m, _ in systems}
        n = max(p.shape[0] for p in probs.values())
        ref = ref_matrix(item["rttm_filepath"], n)
        i0, i1 = int(t0 * 100), int(t1 * 100)
        draw(axes[row], ref[i0:i1], COLORS, t0, t1, "Reference", bold=False)
        title = a.title[ei] if ei < len(a.title) else f"{sset} / {uid}"
        axes[row].set_title(f"({chr(97 + ei)}) {title}", loc="left", fontsize=7.5, pad=2.5, color=FS.INK)
        row += 1
        for m, lab in systems:
            hyp = probs[m] > 0.5
            mp = mapping(ref, hyp)
            rows, cols = [], []
            for r_i in range(ref.shape[1]):
                js = [j for j, rr in mp.items() if rr == r_i]
                rows.append(hyp[:, js[0]] if js else np.zeros(hyp.shape[0], bool))
                cols.append(COLORS[r_i % len(COLORS)])
            for j in range(hyp.shape[1]):
                if j not in mp and hyp[i0:i1, j].any():
                    rows.append(hyp[:, j])
                    cols.append(FS.UNMATCHED)
            mat = np.stack(rows, 1)[i0:i1]
            draw(axes[row], mat, cols, t0, t1, lab, bold=(lab == a.bold))
            row += 1
        for k in range(row - nrow, row - 1):
            axes[k].tick_params(axis="x", labelbottom=False, length=0)
            axes[k].spines["bottom"].set_visible(False)
        last = axes[row - 1]
        last.spines["bottom"].set_color(FS.AXIS)
        last.set_xlabel("Time (s)", labelpad=1.5)
    FS.save(fig, a.out)
    print("saved", a.out, "height_mm", round(fig_h * 25.4, 1))


if __name__ == "__main__":
    main()
