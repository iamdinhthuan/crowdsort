"""DER and speaker-count error as a function of the reference number of speakers.

python fig_nspk.py --systems base:N3D,ft_base_s1:N3D-FT,sfx_full_s1:Proposed --out paper/tex/fig_nspk.pdf
Pools MSDWild few+many (1.04 s) by reference speaker count; prints the table it plots.
"""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = CROWDSORT_ROOT
STYLE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figure.mplstyle")
BINS = [(2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 6, "6"), (7, 7, "7"), (8, 8, "8"), (9, 10, "9-10")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--sets", default="msdwild_few,msdwild_many")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--out", required=True)
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
    plt.style.use(STYLE)
    fig, axs = plt.subplots(1, 2, figsize=(140 / 25.4, 50 / 25.4))
    x = np.arange(len(names))
    for _, lab in systems:
        axs[0].plot(x, [table[lab][n][0] for n in names], label=lab)
        axs[1].plot(x, [table[lab][n][1] for n in names], label=lab)
    for ax, yl in zip(axs, ["DER (%)", "Speaker count MAE"]):
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("Reference number of speakers")
        ax.set_ylabel(yl)
    axs[0].legend(loc="upper left")
    fig.savefig(a.out)
    json.dump(table, open(f"{ROOT}/results/nspk_table.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
