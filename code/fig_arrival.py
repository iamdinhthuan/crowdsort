"""Speaker-level analysis figure: speakers found by arrival rank, and error before and after a return.

python fig_arrival.py --systems N3D,FT-clips,SimConv-MSD,FastMSS,CrowdSort --out paper/tex/fig_arrival.pdf
Reads results/mechanism.json (code/mechanism.py); writes PDF and SVG; prints the plotted values.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/mnt/ssd2/paper/paper_1"
BINS = ["1-2", "3-4", "5-6", "7-+"]
LABELS = {"1-2": "1-2", "3-4": "3-4", "5-6": "5-6", "7-+": "7+"}


def bars(ax, x, vals, systems, w):
    for i, name in enumerate(systems):
        st = FS.SYSTEMS[name]
        ax.bar(x + (i - (len(systems) - 1) / 2) * w, vals[name], w * 0.92, color=st["color"], label=name,
               edgecolor="white", linewidth=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arrival-only", action="store_true", help="omit the return panel promoted to Figure 2")
    a = ap.parse_args()
    FS.apply()
    res = json.load(open(f"{ROOT}/results/mechanism.json"))
    systems = a.systems.split(",")
    ncols = 2 if a.arrival_only else 3
    fig, axs = plt.subplots(1, ncols, figsize=(FS.TEXT_W, 165 / 72),
                            gridspec_kw={"width_ratios": [1, 1] if a.arrival_only else [4, 4, 2.4],
                            "wspace": 0.20 if a.arrival_only else 0.32, "left": 0.07,
                            "right": 0.995, "top": 0.80, "bottom": 0.2})
    w = 0.8 / len(systems)
    for ax, (s, title, lab) in zip(axs[:2], [("msdwild_many", "MSDWild many", "(a)"), ("avaavd_test", "AVA-AVD", "(b)")]):
        x = np.arange(len(BINS))
        vals = {n: [res[s][n][b]["found_pct"] for b in BINS] for n in systems}
        bars(ax, x, vals, systems, w)
        for n in systems:
            print(s, n, "found", vals[n])
        ref = res[s][systems[0]]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LABELS[b]}\nn={ref[b]['n']}" for b in BINS])
        ax.set_xlabel("Arrival rank of the speaker", labelpad=2)
        ax.set_ylim(0, 82)
        ax.set_title(f"{lab} {title}", loc="left", fontsize=8, fontweight="bold", pad=3)
        ax.tick_params(axis="x", length=0)
    axs[0].set_ylabel("Speakers found (%)")
    if not a.arrival_only:
        ax = axs[2]
        x = np.arange(2)
        vals = {n: [res["msdwild_many"][n]["err_before_return_pct"], res["msdwild_many"][n]["err_after_return_pct"]]
                for n in systems}
        bars(ax, x, vals, systems, w)
        for n in systems:
            print("msdwild_many", n, "err other/post-return speech", vals[n])
        ax.set_xticks(x)
        ax.set_xticklabels(["other\nspeech", "post-return\nspeech"])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Own-channel deficit (%)")
        ax.set_ylim(0, 45)
        ax.set_title("(c) MSDWild many", loc="left", fontsize=8, fontweight="bold", pad=3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(systems), bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
               handleheight=0.8, columnspacing=1.4)
    FS.save(fig, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
