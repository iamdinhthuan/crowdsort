"""DER and confusion by position in the recording (results/der_by_time.jsonl), grouped bars."""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
STYLE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figure.mplstyle")
sets = "aishell4_test,ami_test_sdm,ami_test_mhm,ali_test_far,ali_test_near"
series = [("base", "N3D, $M=264$", "#0072B2"), ("base_sc396", "N3D, $M=396$", "#E69F00")]
rows = {}
for l in open(CROWDSORT_ROOT + "/results/der_by_time.jsonl"):
    r = json.loads(l)
    if r["sets"] == sets and len(r["rows"]) == 5:
        rows[r["model"]] = r["rows"]
plt.style.use(STYLE)
fig, ax = plt.subplots(figsize=(140 / 25.4, 55 / 25.4))
labels = ["0-2", "2-5", "5-15", "15-30", ">30"]
x = np.arange(len(labels)); w = 0.38
for i, (m, lab, c) in enumerate(series):
    der = [r["DER"] for r in rows[m]]; conf = [r["conf"] for r in rows[m]]
    ax.bar(x + (i - 0.5) * w, der, w, color=c, label=lab)
    ax.bar(x + (i - 0.5) * w, conf, w, color="none", edgecolor="white", hatch="////", linewidth=0)
    for xi, d in zip(x + (i - 0.5) * w, der):
        ax.text(xi, d + 0.15, f"{d:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Position in the recording (min)"); ax.set_ylabel("DER (%)")
ax.legend(loc="upper center", ncol=2)
ax.set_ylim(0, max(max(r["DER"] for r in rows[m]) for m, _, _ in series) + 2)
fig.savefig(sys.argv[1]); print("saved", sys.argv[1])
