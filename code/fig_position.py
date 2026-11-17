"""DER and confusion by position in the recording (results/der_by_time.jsonl), grouped bars."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csl-figure.mplstyle")
sets = "aishell4_test,ami_test_sdm,ami_test_mhm,ali_test_far,ali_test_near"
series = [("base", "N3D, $M=264$", "#94A3B8"), ("base_sc396", "N3D, $M=396$", "#334155")]
rows = {}
for l in open("/mnt/ssd2/paper/paper_1/results/der_by_time.jsonl"):
    r = json.loads(l)
    if r["sets"] == sets and len(r["rows"]) == 5:
        rows[r["model"]] = r["rows"]
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS
FS.apply()
fig, ax = plt.subplots(figsize=(FS.COL_W, 150 / 72), gridspec_kw={"left": 0.13, "right": 0.99, "top": 0.86, "bottom": 0.26})
labels = ["0–2", "2–5", "5–15", "15–30", ">30"]
# recordings that reach each bin (duration above the bin start)
durs = [json.loads(l)["duration"] / 60 for st in sets.split(",")
        for l in open(f"/mnt/ssd2/paper/paper_1/data/manifests/{st}.jsonl")]
counts = [sum(1 for d in durs if d > lo) for lo in (0, 2, 5, 15, 30)]
print("recordings per bin", counts)
x = np.arange(len(labels)); w = 0.38
for i, (m, lab, c) in enumerate(series):
    der = [r["DER"] for r in rows[m]]; conf = [r["conf"] for r in rows[m]]
    ax.bar(x + (i - 0.5) * w, der, w, color=c, label=lab)
    ax.bar(x + (i - 0.5) * w, conf, w, color="none", edgecolor="white", hatch="////", linewidth=0)
    for xi, d in zip(x + (i - 0.5) * w, der):
        ax.text(xi, d + 0.15, f"{d:.1f}", ha="center", va="bottom", fontsize=7, color=FS.INK)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\nn={c}" for b, c in zip(labels, counts)])
ax.set_xlabel("Position in the recording (min)"); ax.set_ylabel("DER (%)")
ax.tick_params(axis="x", length=0)
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
ax.set_ylim(0, max(max(r["DER"] for r in rows[m]) for m, _, _ in series) + 2)
FS.save(fig, sys.argv[1]); print("saved", sys.argv[1])
