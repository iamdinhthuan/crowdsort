"""Figure 2 as side-by-side speaker-activity matrices.

Rows carry speaker identity; columns carry systems. Colour encodes agreement with
the reference, so a reader need not decode a rainbow of speaker colours. The
underlying thresholded 10 ms predictions, mapping and interval widths are exact.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as FS
from fig_timeline_focus import load, runs, SYSTEMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/activity_matrix"
COLORS = {"Reference": "#334155", "Correct": "#2563EB", "Missing": "#7C8BA1", "Extra": "#D55E00"}
W, H = 522, 294
X0, COL_W, GAP = 29, 87, 12
PITCH = 12.0
NAMES = ["Reference", *[lab for _, lab in SYSTEMS]]


def intervals(activity, d):
    return runs(activity, d["i0"], d["i1"])


def matrix(fig, d, top, letter, title, ticks, event=None):
    # The union over reference AND every system retains all visible errors.
    # Only rows silent in every displayed matrix can be omitted.
    keys = d["active"]
    n = len(keys)
    mat_top = top - 35
    mat_h = n * PITCH
    bottom = mat_top - mat_h
    fig.text(4 / W, top / H, f"({letter})  {title}", va="top", fontsize=9, fontweight="bold")
    end = f"{d['end']:.1f}".rstrip("0").rstrip(".")
    fig.text(518 / W, top / H, f"MSDWild {d['uid']} · {d['start']:g}–{end} s",
             va="top", ha="right", fontsize=7.5, color=FS.SLATE)
    axes = []
    encoded = []
    for si, (name, row) in enumerate(zip(NAMES, d["tracks"])):
        left = X0 + si * (COL_W + GAP)
        fig.text((left + COL_W / 2) / W, (top - 18) / H, name, ha="center", va="top",
                 fontsize=8.5, fontweight="bold" if name in ("Reference", "CrowdSort") else "normal",
                 color=FS.INK)
        ax = fig.add_axes([left / W, bottom / H, COL_W / W, mat_h / H])
        ax.set_xlim(d["start"], d["end"])
        ax.set_ylim(0, n)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelsize=7, length=2, pad=3)
        if ticks[0] == d["start"]:
            ax.get_xticklabels()[0].set_horizontalalignment("left")
        if ticks[-1] == d["end"]:
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.set_yticks(np.arange(n) + .5)
        ax.set_yticklabels([f"S{k+1}" if k < d["n_ref"] else f"U{k-d['n_ref']+1}" for k in keys[::-1]],
                           fontsize=8)
        ax.tick_params(axis="y", length=0, pad=4, labelleft=(si == 0))
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color(FS.AXIS)
        ax.spines["bottom"].set_linewidth(.6)
        ax.grid(False)
        for j in range(n):
            ax.axhspan(j, j + 1, color="#F1F5F9" if j % 2 == 0 else "#FFFFFF", zorder=0)
        for t in ticks:
            ax.axvline(t, color="#E2E8F0", linewidth=.5, zorder=1)
        track_meta = []
        for ki, key in enumerate(keys):
            ref = d["tracks"][0].get(key, np.zeros(len(next(iter(row.values()))), dtype=bool))
            hyp = row.get(key, np.zeros_like(ref))
            states = {"Reference": ref} if si == 0 else {
                "Correct": ref & hyp,
                "Missing": ref & ~hyp,
                "Extra": ~ref & hyp,
            }
            counts = {}
            for state, mask in states.items():
                spans = intervals(mask, d)
                if spans:
                    ax.broken_barh(spans, (n - ki - .90, .80), facecolors=COLORS[state],
                                   edgecolors="none", zorder=3)
                count = int(mask[d["i0"]:d["i1"]].sum())
                assert round(sum(duration * 100 for _, duration in spans)) == count
                counts[state] = count
            if si:
                assert counts["Correct"] + counts["Missing"] == int(ref[d["i0"]:d["i1"]].sum())
                assert counts["Correct"] + counts["Extra"] == int(hyp[d["i0"]:d["i1"]].sum())
            track_meta.append({"speaker": key + 1, "state_frames": counts})
        if event is not None:
            ax.axvline(event, color="#64748B", linewidth=.7, linestyle=(0, (2, 2)), zorder=4)
        axes.append(ax)
        encoded.append({"system": name, "speaker_states": track_meta})
    fig.text(.515, (bottom - 22) / H, "Time (s)", ha="center", va="top", fontsize=7.5, color=FS.SLATE)
    return {"set": d["sset"], "uid": d["uid"], "range_s": [d["start"], d["end"]],
            "total_reference_speakers": d["n_ref"], "keys": keys, "onset_s": event,
            "mapping": d["mapping"], "encoded_frames": encoded}


def main():
    FS.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
    a = load("msdwild_many", "01151", 435, 455)
    b = load("msdwild_many", "02311", 0, 54)
    meta = [matrix(fig, a, 289, "a", "Delayed allocation of S5", [435, 445, 455], 440.315),
            matrix(fig, b, 170, "b", "Failure: speaker misassignment", [0, 25, 50])]
    # One shared legend for agreement, not a second encoding of speaker identity.
    x = 109
    for label, width in [("Reference", 91), ("Correct", 77), ("Missing", 78), ("Extra", 60)]:
        fig.add_artist(Rectangle((x / W, 7 / H), 10 / W, 5 / H, transform=fig.transFigure,
                                 facecolor=COLORS[label], edgecolor="none"))
        fig.text((x + 14) / W, 9.5 / H, label, va="center", fontsize=7.5, color=FS.INK)
        x += width
    fig.savefig(OUT / "fig_activity_matrix.pdf")
    fig.savefig(OUT / "fig_activity_matrix.svg")
    fig.savefig(OUT / "fig_activity_matrix.png", dpi=200)
    (OUT / "plot_metadata.json").write_text(json.dumps({"width_pt": W, "height_pt": H,
        "threshold": ">0.5", "frame_seconds": .01, "colors": COLORS,
        "semantics": "Speaker-wise binary activity agreement, not DER component attribution",
        "panels": meta}, indent=2) + "\n")
    print(f"Saved activity matrix: {W} × {H} pt; exact per-speaker frame partitions validated.")


if __name__ == "__main__":
    main()
