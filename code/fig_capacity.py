import os
"""DER versus speaker-cache size at inference, with throughput, for one or more models.

python fig_capacity.py --series base:N3D:132,264,330,396,462,528 --out paper/tex/fig_capacity.pdf
Model names per size: <model> for 264 and <model>_sc<size> otherwise (base_sc396 ...), or <model>_c<size>
when --suffix _c is given.
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/mnt/ssd2/paper/paper_1"
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csl-figure.mplstyle")
SETS = {"ami_test_sdm": "AMI SDM", "aishell4_test": "AISHELL-4", "msdwild_many": "MSDWild many",
        "pooled": "Pooled (7 sets)"}
ALL = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few", "msdwild_many"]


def pooled_der(model, profile="1.04"):
    tot = err = 0.0
    for s in ALL:
        pf = json.load(open(f"{ROOT}/exp/{model}/{s}/{profile}/_perfile_thr.json"))
        for r in pf.values():
            tot += r["total"]
            err += r["miss"] + r["fa"] + r["conf"]
    return 100 * err / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, help="model:label:sizes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    for l in open(f"{ROOT}/results/results.jsonl"):
        r = json.loads(l)
        R[(r["model"], r["set"], r["profile"], r["decode"])] = r["DER"]
    speed = {}
    for l in open(f"{ROOT}/results/speed.jsonl"):
        r = json.loads(l)
        if r["name"] == "N3D" and r["profile"] == "1.04" and r["batch"] == 16:
            speed[r.get("spkcache", 264)] = r["rtfx"]
    plt.style.use(STYLE)
    fig, axs = plt.subplots(1, len(SETS), figsize=(190 / 25.4, 50 / 25.4))
    for spec in a.series:
        model, label, sizes = spec.split(":")
        sizes = [int(x) for x in sizes.split(",")]
        names = [model if c == 264 else f"{model}_sc{c}" for c in sizes]
        for ax, (s, title) in zip(axs, SETS.items()):
            if s == "pooled":
                ys = [pooled_der(n) for n in names]
            else:
                ys = [R.get((n, s, "1.04", "thr"), np.nan) for n in names]
            ax.plot(sizes, ys, label=label)
            ax.set_title(title)
            ax.set_xlabel("Cache size $M$ (frames)")
            print(label, s, [round(y, 2) for y in ys])
    axs[0].set_ylabel("DER (%)")
    if speed:
        ax2 = axs[-1].twinx()
        ks = sorted(speed)
        ax2.plot(ks, [speed[k] for k in ks], color="0.5", linestyle="--", marker="x", label="RTFx")
        ax2.set_ylabel("RTFx (batch 16)", color="0.4")
        ax2.spines["right"].set_visible(True)
    axs[0].legend(loc="upper right")
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
