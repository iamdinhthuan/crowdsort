"""Readable, data-derived system-output comparisons at the manuscript print size.

The main figure contains the existing rule-selected improvement and failure cases.
Full context and the median examples are exported as separate supplementary panels.
All systems use saved seed-1 probabilities, threshold 0.5, and the original full-
recording Hungarian channel mapping. No duration is enlarged for visibility.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as FS
from fig_timeline import mapping, ref_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/timelines"
SYSTEMS = [("base", "N3D"), ("c3_noconcat", "FT-clips"),
           ("c3_ci_s1", "CrowdSort"), ("c3_cip_s1", "CrowdSort-P")]
W = 522
LEFT = 76
PLOT_W = 379
METRIC_X = 511


def load(sset, uid, start, end):
    manifest = ROOT / "data/manifests" / f"{sset}.jsonl"
    items = [json.loads(line) for line in manifest.open()]
    item = next(it for it in items if Path(it["audio_filepath"]).stem == uid)
    probs = {model: np.load(ROOT / "exp" / model / sset / "1.04" / f"{uid}.npz")["probs"]
             for model, _ in SYSTEMS}
    n = max(len(p) for p in probs.values())
    reference = ref_matrix(item["rttm_filepath"], n)
    end = min(end, item["duration"])
    i0, i1 = int(start * 100), min(int(end * 100), n)
    tracks = [{k: reference[:, k] for k in range(reference.shape[1])}]
    scores = [None]
    maps = {}
    for model, _ in SYSTEMS:
        hyp = probs[model] > .5
        mp = mapping(reference, hyp)
        maps[model] = {int(k): int(v) for k, v in mp.items()}
        row = {k: np.zeros(n, dtype=bool) for k in range(reference.shape[1])}
        u = 0
        for j in range(hyp.shape[1]):
            k = mp.get(j)
            if k is None:
                k = reference.shape[1] + u
                u += 1
            dest = np.zeros(n, dtype=bool)
            dest[:len(hyp)] = hyp[:, j]
            row[k] = dest
        tracks.append(row)
        stat = json.load((ROOT / "exp" / model / sset / "1.04" / "_perfile_thr.json").open())[uid]
        scores.append(100 * sum(stat[k] for k in ("miss", "fa", "conf")) / stat["total"])
    active = sorted({k for row in tracks for k, activity in row.items() if activity[i0:i1].any()})
    return {"sset": sset, "uid": uid, "start": start, "end": end, "i0": i0, "i1": i1,
            "tracks": tracks, "scores": scores, "active": active, "n_ref": reference.shape[1],
            "mapping": maps}


def runs(activity, i0, i1):
    x = np.r_[False, activity[i0:i1], False].astype(np.int8)
    d = np.diff(x)
    return [(i0 / 100 + b / 100, (e - b) / 100)
            for b, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def panel(fig, d, top, letter, title, ticks, event=None):
    """One panel with physical speaker-lane sizes and one shared time axis."""
    lane_pitch = 2.6
    row_h = max(22, len(d["active"]) * lane_pitch + 4)
    plot_h = row_h * 5
    plot_top = top - 27
    bottom = plot_top - plot_h
    total_h = fig.get_figheight() * 72
    fig.text(5 / W, top / total_h, f"({letter})  {title}", fontsize=9, fontweight="bold", va="top")
    corpus = "MSDWild" if d["sset"] == "msdwild_many" else "AVA-AVD"
    time_label = lambda t: f"{t:.1f}".rstrip("0").rstrip(".")
    info = f"{corpus} / {d['uid']}  ·  {time_label(d['start'])}–{time_label(d['end'])} s"
    fig.text(5 / W, (top - 13) / total_h, info, fontsize=7.5, color=FS.SLATE, va="top")
    fig.text(METRIC_X / W, (plot_top + 6) / total_h, "DER (%)", fontsize=7.5,
             color=FS.SLATE, ha="right", va="bottom")
    ax = fig.add_axes([LEFT / W, bottom / total_h, PLOT_W / W, plot_h / total_h])
    ax.set_xlim(d["start"], d["end"])
    ax.set_ylim(0, plot_h)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=7.5, length=2, pad=3)
    ax.grid(axis="x", color=FS.GRID, linewidth=.5)
    ax.set_axisbelow(True)
    for name in ("left", "top", "right"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_linewidth(.65)
    labels = ["Reference", *[label for _, label in SYSTEMS]]
    for ri, (row, label, der) in enumerate(zip(d["tracks"], labels, d["scores"])):
        y0 = plot_h - (ri + 1) * row_h
        bg = "#EFF6FF" if label == "CrowdSort" else "#F8FAFC" if ri % 2 == 0 else "#FFFFFF"
        ax.axhspan(y0, y0 + row_h, color=bg, zorder=0)
        for ki, key in enumerate(d["active"]):
            if key not in row:
                continue
            y = y0 + (row_h - len(d["active"]) * lane_pitch) / 2 + (len(d["active"]) - ki - 1) * lane_pitch
            color = FS.SPEAKERS[key] if key < d["n_ref"] else FS.UNMATCHED
            intervals = runs(row[key], d["i0"], d["i1"])
            if intervals:
                ax.broken_barh(intervals, (y, 1.95), facecolors=color, edgecolors="none", zorder=3)
        center = (bottom + y0 + row_h / 2) / total_h
        proposed = label == "CrowdSort"
        fig.text((LEFT - 9) / W, center, label, ha="right", va="center", fontsize=8,
                 fontweight="bold" if proposed else "normal", color=FS.INK)
        fig.text(METRIC_X / W, center, "—" if der is None else f"{der:.1f}",
                 ha="right", va="center", fontsize=8.5, color=FS.INK,
                 fontweight="bold" if proposed else "normal")
    if event is not None:
        ax.axvline(event, color="#DC2626", linewidth=.7, linestyle=(0, (3, 2)), zorder=2)
        # An observed reference onset, not an automatically inferred model event.
    ax.set_xlabel("Time (s)", fontsize=7.5, labelpad=3)
    return bottom - 27


def legend(fig, y):
    h = fig.get_figheight() * 72
    fig.text(5 / W, y / h, "Speakers:", fontsize=7, color=FS.SLATE, va="center")
    for k in range(9):
        x = 49 + k * 37
        fig.add_artist(Rectangle((x / W, (y - 2) / h), 8 / W, 4 / h,
                                 transform=fig.transFigure, facecolor=FS.SPEAKERS[k], edgecolor="none"))
        fig.text((x + 11) / W, y / h, f"S{k+1}", fontsize=7, va="center", color=FS.SLATE)
    x = 392
    fig.add_artist(Rectangle((x / W, (y - 2) / h), 8 / W, 4 / h,
                             transform=fig.transFigure, facecolor=FS.UNMATCHED, edgecolor="none"))
    fig.text((x + 11) / W, y / h, "Unmatched", fontsize=7, va="center", color=FS.SLATE)


def save_pair(path, specs):
    heights = [max(22, len(d["active"]) * 2.6 + 4) * 5 + 27 + 27 for d, *_ in specs]
    total_h = sum(heights) + 10 * (len(specs) - 1) + 24
    fig = plt.figure(figsize=(W / 72, total_h / 72), facecolor="white")
    top = total_h - 4
    for i, (d, title, ticks, event) in enumerate(specs):
        top = panel(fig, d, top, chr(ord('a') + i), title, ticks, event) - 10
    legend(fig, 8)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
    return {"file": str(path.relative_to(ROOT)), "width_pt": W, "height_pt": round(total_h, 2),
            "panels": [{k: v for k, v in d.items() if k not in ("tracks",)} for d, *_ in specs]}


def main():
    FS.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    success = load("msdwild_many", "01151", 425, 550)
    failure = load("msdwild_many", "02311", 0, 54)
    full = load("msdwild_many", "01151", 0, 1200)
    median = load("msdwild_many", "02795", 0, 23)
    ava = load("avaavd_test", "PmElx9ZVByw_c_01", 0, 300)
    results = []
    results.append(save_pair(OUT / "fig_timeline_comparison.pdf", [
        (success, "Late speaker allocation", [425, 450, 475, 500, 525, 550], 440.315),
        (failure, "Failure: fragmented speaker assignment", [0, 10, 20, 30, 40, 50], None),
    ]))
    results.append(save_pair(OUT / "fig_timeline_context.pdf", [
        (full, "Full recording of the improvement case", [0, 200, 400, 600, 800, 1000, 1200], None),
        (median, "Median-change MSDWild example", [0, 5, 10, 15, 20], None),
    ]))
    results.append(save_pair(OUT / "fig_timeline_transfer.pdf", [
        (ava, "Upper-median AVA-AVD example", [0, 50, 100, 150, 200, 250, 300], None),
    ]))
    (OUT / "plot_metadata.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(r['file'], f"{r['width_pt']} × {r['height_pt']} pt")


if __name__ == "__main__":
    main()
