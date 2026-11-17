"""A compact Figure 2 candidate based on the existing corpus-level return analysis.

No new experiment or inferential test: plots the already reported, rounded,
reference-speech-weighted summaries from results/mechanism.json.
"""
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle as FS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/design_review"


def main():
    FS.apply()
    data = json.loads((ROOT / "results/mechanism.json").read_text())["msdwild_many"]
    specs = [("FTclips", "FT-clips", FS.SYSTEMS["FT-clips"]["color"]),
             ("CrowdSort", "CrowdSort", FS.SYSTEMS["CrowdSort"]["color"])]
    fields = ["err_before_return_pct", "err_after_return_pct"]
    fig, ax = plt.subplots(figsize=(252 / 72, 180 / 72))
    fig.subplots_adjust(left=.20, right=.97, bottom=.26, top=.84)
    x = np.arange(2)
    for k, (key, label, color) in enumerate(specs):
        values = [data[key][f] for f in fields]
        bars = ax.bar(x + (k - .5) * .30, values, width=.26, color=color, label=label,
                      edgecolor="none")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Own-channel coverage deficit (%)", fontsize=8, labelpad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Other speech", "Post-return speech"], fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_title("MSDWild many · 3 seeds", loc="left", fontsize=9, fontweight="bold", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .025), frameon=False,
               ncol=2, fontsize=8, handlelength=1.1, columnspacing=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "aggregate_return.pdf")
    fig.savefig(OUT / "aggregate_return.svg")
    fig.savefig(OUT / "aggregate_return.png", dpi=250)
    (OUT / "aggregate_return_metadata.json").write_text(json.dumps({
        "source": "results/mechanism.json", "set": "msdwild_many", "recordings": 177,
        "systems": {key: {f: data[key][f] for f in fields} for key, _, _ in specs},
        "weighting": "reference speech; three seeds pooled",
        "other_speech": "before a speaker's first qualifying return plus speakers who never return",
        "after_return": "all speech from the speaker's first resumption after >=10 s of own silence",
        "metric": "reference speech not covered by its optimally assigned channel, not DER",
        "descriptive_only": True, "width_pt": 252, "height_pt": 180,
    }, indent=2) + "\n")
    print("Aggregate alternative saved: 252 × 180 pt.")


if __name__ == "__main__":
    main()
