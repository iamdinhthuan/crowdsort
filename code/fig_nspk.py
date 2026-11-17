"""DER and speaker-count error as a function of the reference number of speakers.

python fig_nspk.py --systems base:N3D,ft_base_s1:N3D-FT,sfx_full_s1:Proposed --out paper/tex/fig_nspk.pdf
Pools MSDWild few+many (1.04 s) by reference speaker count; prints the table it plots.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/mnt/ssd2/paper/paper_1"
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csl-figure.mplstyle")
BINS = [(2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 6, "6"), (7, 7, "7"), (8, 8, "8"), (9, 10, "9-10")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--sets", default="msdwild_few,msdwild_many")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width-mm", type=float, default=140)
    ap.add_argument("--height-mm", type=float, default=50)
    a = ap.parse_args()
    systems = [s.split(":") for s in a.systems.split(",")]
    table = {}
    for m, lab in systems:
        agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # total, err, |count err|, n
        for s in a.sets.split(","):
            pf = json.load(open(f"{ROOT}/exp/{m}/{s}/{a.profile}/_perfile_thr.json"))
            for uid, r in pf.items():
                for lo, hi, name in BINS:
                    if lo <= r["n_ref"] <= hi:
                        g = agg[name]
                        g[0] += r["total"]
                        g[1] += r["miss"] + r["fa"] + r["conf"]
                        g[2] += abs(r["n_hyp"] - r["n_ref"])
                        g[3] += 1
        table[lab] = {k: (100 * v[1] / v[0], v[2] / v[3], v[3]) for k, v in agg.items()}
    names = [b[2] for b in BINS if any(b[2] in t for t in table.values())]
    print("n_spk".ljust(8) + "".join(l.rjust(16) for _, l in systems))
    for n in names:
        print(n.ljust(8) + "".join(
            (f"{table[l][n][0]:6.2f}/{table[l][n][1]:4.2f}" if n in table[l] else "-").rjust(16) for _, l in systems)
            + f"   n={table[systems[0][1]][n][2]}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figstyle as FS
    FS.apply()
    fig, axs = plt.subplots(1, 2, figsize=(FS.COL_W, a.height_mm / 25.4),
                            gridspec_kw={"wspace": 0.42, "left": 0.13, "right": 0.99, "top": 0.80, "bottom": 0.2})
    x = np.arange(len(names))
    for _, lab in systems:
        st = FS.SYSTEMS[lab]
        kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], label=lab, markersize=3.2, linewidth=1.0)
        axs[0].plot(x, [table[lab][n][0] for n in names], **kw)
        axs[1].plot(x, [table[lab][n][1] for n in names], **kw)
    for ax, yl, t in zip(axs, ["DER (%)", "Speaker count MAE"], ["(a)", "(b)"]):
        ax.set_xticks(x)
        ax.set_xticklabels(["9+" if n == "9-10" else n for n in names])
        ax.set_xlabel("Number of speakers", labelpad=1.5)
        ax.set_ylabel(yl, labelpad=1.5)
        ax.set_title(t, loc="left", fontsize=8, fontweight="bold", pad=2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.0)
    FS.save(fig, a.out)
    json.dump(table, open(f"{ROOT}/results/nspk_table.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
