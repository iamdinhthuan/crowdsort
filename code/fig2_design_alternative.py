"""One focused, reproducible Figure 2 prototype; never updates the manuscript.

Run from any directory with PYTHONDONTWRITEBYTECODE=1. Only outputs beneath
paper/figures/design_review/fig2_alternative* are written. Uses the existing
validated thresholding and full-recording mapping.
No inference, smoothing, probability traces, or synthetic audio illustrations.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

import figstyle as FS
from fig_timeline_focus import load, runs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/design_review"
STEM = "fig2_alternative"
W, H = 522, 410
FT, CS, REF = "#B96B09", FS.SYSTEMS["CrowdSort"]["color"], FS.AXIS
SELECTED = [(2, "c3_noconcat", "FT-clips"), (3, "c3_ci_s1", "CrowdSort")]
RETURN_S = 440.315  # Checked against the RTTM below, not a model estimate.


def text(fig, x, top, s, size=8, color=FS.INK, **kw):
    """Position in points, measured from the upper left of the fixed canvas."""
    return fig.text(x / W, 1 - top / H, s, fontsize=size, color=color,
                    va=kw.pop("va", "top"), **kw)


def recording_item(uid):
    with (ROOT / "data/manifests/msdwild_many.jsonl").open() as f:
        return next(it for line in f if Path((it := json.loads(line))["audio_filepath"]).stem == uid)


def raw_speaker_intervals(uid, speaker_index):
    item = recording_item(uid)
    segments = []
    for line in Path(item["rttm_filepath"]).read_text().splitlines():
        fields = line.split()
        segments.append((float(fields[3]), float(fields[4]), fields[7]))
    first = {}
    for start, _, identity in segments:
        first[identity] = min(first.get(identity, float("inf")), start)
    speaker = sorted(first, key=first.get)[speaker_index]
    return [(s, duration) for s, duration, identity in segments if identity == speaker]


def first_activation(row, speaker, start):
    indices = np.flatnonzero(row[speaker][int(start * 100):])
    return (int(start * 100) + int(indices[0])) / 100 if len(indices) else None


def frame_counts(d):
    result = []
    for k in range(d["n_ref"]):
        ref = d["tracks"][0][k][d["i0"]:d["i1"]]
        n = int(ref.sum())
        systems = {}
        for index, model, label in SELECTED:
            hyp = d["tracks"][index][k][d["i0"]:d["i1"]]
            both = int((ref & hyp).sum())
            missing = int((ref & ~hyp).sum())
            extra = int((~ref & hyp).sum())
            assert both + missing == n
            assert both + extra == int(hyp.sum())
            systems[label] = dict(model_id=model, both_frames=both,
                                  missing_frames=missing, extra_frames=extra,
                                  correctly_assigned_s=both / 100,
                                  reference_coverage_percent=100 * both / n if n else None)
        result.append(dict(speaker=f"S{k+1}", reference_frames=n,
                           reference_s=n / 100, systems=systems))
    return result


def return_panel(fig, d):
    text(fig, 12, 11, "(a) Recovery of a returning speaker", size=10, weight="bold")
    text(fig, 510, 12, "01151 · 435–455 s excerpt", size=8.5, ha="right", color=FS.SLATE)
    text(fig, 12, 29,
         f"Full-recording DER:  FT-clips {d['scores'][2]:.2f}%  →  CrowdSort {d['scores'][3]:.2f}%",
         size=8.5)

    left, right, top, bottom = 115, 510, 60, 160
    ax = fig.add_axes([left / W, (H - bottom) / H, (right - left) / W, (bottom - top) / H])
    ax.set_xlim(435, 455)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.set_xticks([435, 440, 445, 450, 455])
    ax.tick_params(axis="x", labelsize=8, length=3, pad=4)
    ax.set_xlabel("Time in recording (s)", fontsize=8, labelpad=6)
    ax.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(FS.GRID)
    for tick in [435, 440, 445, 450, 455]:
        ax.axvline(tick, color="#EBEFF4", linewidth=.45, zorder=0)

    # Two FT-clips channels belong to one comparison, not two different systems.
    ax.add_patch(Rectangle((435, 94), 20, 43, color="#FBF6EF", lw=0, zorder=0))
    rows = [("Reference: S5", 0, 4, 82, REF),
            ("FT-clips: S2", 2, 1, 106, FT),
            ("FT-clips: S5", 2, 4, 128, FT),
            ("CrowdSort: S5", 3, 4, 151, CS)]
    exported = []
    for label, index, key, y, color in rows:
        mask = d["tracks"][index][key]
        spans = runs(mask, d["i0"], d["i1"])
        assert round(100 * sum(duration for _, duration in spans)) == int(mask[d["i0"]:d["i1"]].sum())
        ax.hlines(y, 435, 455, color="#DEE4EC", linewidth=.55, zorder=1)
        if spans:
            ax.broken_barh(spans, (y - 4, 8), facecolors=color, edgecolors="none", zorder=3)
        text(fig, 105, y, label, size=8.5, ha="right", va="center",
             color=color if index else REF, weight="bold" if index == 3 else "normal")
        exported.append(dict(label=label, model_index=index, speaker=key+1,
                             intervals_start_duration_s=spans))

    assert not d["tracks"][0][1][d["i0"]:d["i1"]].any()  # S2 reference is silent.
    assert not d["tracks"][2][4][d["i0"]:d["i1"]].any()
    onset = first_activation(d["tracks"][3], 4, RETURN_S)
    assert onset == 442.08
    assert any(abs(s - RETURN_S) < 1e-9 for s, _ in raw_speaker_intervals("01151", 4))
    ax.vlines(RETURN_S, 73, 157, color=REF, linestyles=(0, (2, 2)), linewidth=.7, zorder=4)
    ax.vlines(onset, 144, 158, color=CS, linewidth=.85, zorder=4)
    ax.plot([RETURN_S, RETURN_S, onset, onset], [71, 67, 67, 71],
            color=CS, lw=.9, zorder=4)
    # Decimal subtraction avoids binary float formatting 1.765 as 1.76.
    elapsed = Decimal(str(onset)) - Decimal(str(RETURN_S))
    elapsed_label = elapsed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    ax.text(onset + .28, 67, f"{elapsed_label} s", color=CS, va="center", fontsize=8)
    text(fig, left, 47, f"S5 resumes at {RETURN_S:.3f} s", size=8)
    text(fig, right, 47, f"CrowdSort S5 starts at {onset:.2f} s", size=8, ha="right", color=CS)
    ax.text(443.7, 128, "No S5 output in this window", va="center", ha="left", color=FT,
            fontsize=8, bbox=dict(facecolor="#FBF6EF", edgecolor="none", pad=2))
    text(fig, 12, 193, "S2 is reference-silent; some S5 speech is assigned to S2. Only named channels are shown.", size=8)
    return dict(uid=d["uid"], excerpt_s=[d["start"], d["end"]],
                reference_s5_resumption_rttm_s=RETURN_S,
                reference_s5_resumption_frame_s=440.32,
                crowdsort_s5_activation_s=onset, elapsed_s=float(elapsed),
                elapsed_display_s=str(elapsed_label),
                plotted_tracks=exported)


def failure_panel(fig, d, counts):
    text(fig, 12, 217, "(b) Failure: less speech retains the correct identity", size=10, weight="bold")
    text(fig, 510, 218, "02311 · full 53.499 s", size=8.5, ha="right", color=FS.SLATE)
    text(fig, 12, 235,
         f"Full-recording DER:  FT-clips {d['scores'][2]:.2f}%  →  CrowdSort {d['scores'][3]:.2f}%",
         size=8.5)
    text(fig, 12, 256, "Reference speech", size=8, color=FS.SLATE)
    text(fig, 425, 252, "Correct speech", size=8, color=FS.SLATE)
    text(fig, 425, 262, "lost vs. FT-clips", size=8, color=FS.SLATE)

    left, right, top, bottom = 101, 385, 276, 365
    ax = fig.add_axes([left/W, (H-bottom)/H, (right-left)/W, (bottom-top)/H])
    ax.set_xlim(-2, 102)
    ax.set_ylim(6.5, -.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=8, length=3, pad=4)
    ax.set_xlabel("Reference speech assigned to the correct speaker (%)", fontsize=8, labelpad=5)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(FS.GRID)
    ax.grid(False)
    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color=FS.GRID, lw=.5, zorder=0)
    for i, speaker in enumerate(counts):
        a, b = [speaker["systems"][label] for label in ["FT-clips", "CrowdSort"]]
        x0, x1 = a["reference_coverage_percent"], b["reference_coverage_percent"]
        important = speaker["speaker"] in ("S5", "S6")
        if important:
            ax.axhspan(i-.45, i+.45, color="#EEF4FC", zorder=0)
        ax.plot([x0, x1], [i, i], color="#9DAABE", lw=1.3, zorder=1)
        ax.plot(x0, i, marker="s", markersize=6.5, markerfacecolor="white",
                markeredgecolor=FT, markeredgewidth=1.1, linestyle="none", zorder=3)
        ax.plot(x1, i, marker="o", markersize=4, color=CS, linestyle="none", zorder=4)
        y = top + (i+.5) * (bottom-top) / 7
        text(fig, 91, y, f"{speaker['speaker']} · {speaker['reference_s']:.2f} s", size=8,
             ha="right", va="center", weight="bold" if important else "normal")
        if important:
            lost = a["correctly_assigned_s"] - b["correctly_assigned_s"]
            text(fig, 450, y, f"−{lost:.2f} s", size=9, weight="bold", color=CS,
                 ha="center", va="center")

    # A ring surrounds a filled dot when both systems are at zero; neither is hidden.
    handles = [Line2D([], [], marker="s", ms=6.5, mfc="white", mec=FT, mew=1.1,
                      ls="none", label="FT-clips"),
               Line2D([], [], marker="o", ms=4, color=CS, ls="none", label="CrowdSort")]
    legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.07),
                       ncol=2, frameon=False, fontsize=8.5, handletextpad=.45,
                       columnspacing=1.8, borderaxespad=0)
    for label in legend.get_texts():
        label.set_color(FT if label.get_text() == "FT-clips" else CS)
    text(fig, 12, 399, "Correct activity excludes false alarms; these values do not decompose DER.", size=8)


def provenance(ds):
    sources = []
    for d in ds:
        item = recording_item(d["uid"])
        paths = [ROOT/"data/manifests/msdwild_many.jsonl", Path(item["rttm_filepath"])]
        for _, model, _ in SELECTED:
            base = ROOT / "exp" / model / "msdwild_many/1.04"
            paths.extend([base / f"{d['uid']}.npz", base / "_perfile_thr.json"])
        sources.append(dict(uid=d["uid"], full_recording_duration_s=item["duration"],
                            full_recording_der_percent={label: d["scores"][i] for i, _, label in SELECTED},
                            mapping=d["mapping"],
                            files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                   for p in paths]))
    return sources


def review_text(counts):
    rows = []
    for s in counts:
        ft, cs = [s["systems"][label] for label in ["FT-clips", "CrowdSort"]]
        rows.append(f"| {s['speaker']} | {s['reference_s']:.2f} | "
                    f"{ft['correctly_assigned_s']:.2f} | {cs['correctly_assigned_s']:.2f} | "
                    f"{ft['reference_coverage_percent']:.2f}% | {cs['reference_coverage_percent']:.2f}% |")
    return """# Figure 2: focused return event and a speaker-level failure

## Diagnosis

The rainbow timelines require repeated speaker-colour lookup; the subsequent five-column matrices remove that lookup but divide the useful time axis into five narrow strips. Both leave the reader to infer the important identity error from many disconnected marks. Equal space for every model and speaker makes sparse, irrelevant rows as prominent as the event. At manuscript scale the original small labels and thin brief turns further weaken the hierarchy. Colour agreement also invites interpreting speaker-wise missing/extra cells as DER components, which they are not.

## One prototype

The upper panel is one shared, long time axis with four explicitly named tracks. Two adjacent FT-clips tracks answer the actual identity question: S2 is predicted while the correct S5 channel stays inactive. CrowdSort's S5 onset and the reference return are explicitly timed. The lower panel changes visual grammar to a connected-dot comparison of correctly assigned reference speech for all seven failure-case speakers. It preserves the counterexample and gives S5/S6 losses a readable numerical scale without another grid of activity marks. Model colours remain stable, with direct channel labels and marker shapes as redundant cues.

## Selection and omissions

Cases remain the existing rule-selected extremes: 01151 is the largest full-recording DER improvement and 02311 the largest degradation from FT-clips to CrowdSort among the 41 MSDWild-many recordings with at least seven speakers. These ranks are inherited from the already validated selection, not reselected to favour this prototype. Neither case is described as typical.

The improvement excerpt stays at 435–455 s. Only reference S5, FT-clips S2/S5, and CrowdSort S5 are drawn. S2 is reference-silent throughout the excerpt. The S2 trace demonstrates one concrete wrong-channel assignment; FT-clips also emits S3, and CrowdSort has brief other-channel errors, all deliberately omitted. The S2 trace is not a complete attribution of all S5 errors. Reference activity of other speakers is omitted. No time interval or brief output is enlarged, interpolated, thresholded again, or smoothed.

The failure summary uses the full 53.499 s recording and includes every reference speaker S1–S7, including near-zero rows. It shows reference duration as well as the fraction jointly active in reference and the speaker's mapped prediction. No speaker or turn is selected within that failure recording. Temporal order, erroneous extra output, and destinations of wrong identities are omitted by the summary.

N3D and CrowdSort-P, full improvement-recording timelines, and the median/transfer examples are omitted from this main prototype. Those can stay in the existing supplementary material. Only FT-clips (c3_noconcat) and CrowdSort (c3_ci_s1) are drawn. All saved arrays and full-recording mappings still come from the existing loader; its base and c3_cip_s1 loads do not become visual evidence. Fine-tuned seed 1, profile 1.04 s, p > 0.5, and 10 ms frames are unchanged.

## Evidence and limits

In 01151 the exact RTTM return is 440.315 s, drawn against the 10 ms reference trace whose first active frame starts at 440.32 s. The first saved CrowdSort S5 output is 442.08 s; elapsed time is 1.765 s (displayed as 1.77 s), measured on the output timeline. This is not a separate measurement of wall-clock end-to-end latency. FT-clips has zero S5 activity in the excerpt. Full-recording DER is 34.8571% versus 9.6675%; these DER values are context for the recording, not the excerpt's DER. The cropped trace does not establish persistence to the recording end, the eventual FT-clips onset, or that every local CrowdSort output is correct.

In 02311 whole-recording DER increases from 47.8848% to 62.1775%. S5 loses 4.00 s and S6 loses 1.55 s of jointly active reference/prediction time. S4 also loses 1.20 s; S2 loses 0.42 s. All appear in the connected-dot plot, so the failure is not framed as an S2-only issue. Coverage does not penalize a model for false positives or simultaneous wrong channels, ignores the scoring collar in the DER evaluation, and is not a DER decomposition. These selected examples illustrate observed output behaviour, not typical gains, cache states, causal effects of an individual synthesis operator, or statistical generality.

## Candid main-text judgment

This is a better qualitative illustration than either rejected version: the return timing and wrong identity can be decoded directly, and the failure has explicit S5/S6 evidence rather than an undirected field of errors. Its scientific contribution is showing the particular form of the error, which an aggregate post-return coverage chart cannot display. Its cost is a 410 pt-high figure for two deliberately extreme recordings, one crop, selected channels, and a coverage statistic that is not DER. The failure plot is a within-recording summary across speakers, not an aggregate return-event analysis across recordings.

My recommendation is to prefer the separately produced aggregate post-return coverage candidate for the main text if its selection and denominators are sound, and put this case illustration in the supplement. The case illustration merits main-text space only if readers specifically need to see how a wrong existing identity differs from the newly active S5 channel; even then, a compact return-event inset would be more economical than this full two-case prototype. I have not inspected the other candidate and do not claim that its physical layout or scientific estimand is validated here. No aggregate_return.* files are read or written by this script.

## Exact failure values

| Speaker | Reference s | FT-clips correct s | CrowdSort correct s | FT-clips coverage | CrowdSort coverage |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## Suggested standalone caption (not applied to the paper)

**Targeted output comparison on rule-selected improvement and failure recordings.** (a) Return of S5 in MSDWild 01151. FT-clips emits S2 during some reference S5 speech and has no S5 output in this 435–455 s excerpt; CrowdSort first activates the mapped S5 channel at 442.08 s, 1.77 s after the reference return at 440.315 s. Only the named channels are shown; other outputs and errors are omitted. (b) Full MSDWild 02311: connected squares (FT-clips) and circles (CrowdSort) show the percentage of each reference speaker's speech that also activates that speaker's mapped output. Labels give reference speech durations; highlighted S5/S6 losses are in seconds. Correct activity excludes false alarms and does not decompose DER. Whole-recording DER is reported separately above each panel. These are the largest gain and largest degradation among the 41 MSDWild-many recordings with at least seven speakers, not representative examples. Outputs use p > 0.5, 10 ms frames, full-recording channel mapping, seed 1 and profile 1.04 s.

## Reproduction and physical checks

Run `PYTHONDONTWRITEBYTECODE=1 python code/fig2_design_alternative.py`. The canvas is exactly 522 × 410 pt. All figure text is at least 8 pt. PDF and SVG use vector shapes. PDF embeds TrueType fonts; SVG retains text elements and depends on compatible installed fonts. The PNG is 300 dpi. The script asserts interval/frame agreement, verifies the exact RTTM return and the saved output onset, checks all text against canvas boundaries, and writes source hashes, mappings and exact frame counts in the companion JSON. The PNG must also be inspected visually. No manuscript, existing generator or existing PDF is modified.
"""


PHILOSOPHY = """# Evidence First

Space follows the reader's question. A long shared axis establishes a single event; a separate comparison isolates its limit. Each grouping is meticulously aligned so its structure can be understood before any colour is decoded.

Colour is restrained and stable. Neutral reference marks establish what happened; a warm square and a blue circle distinguish two alternatives with redundant shape cues. White space separates meanings rather than decorating them. This is the product of careful visual editing.

Scale is physical. Type remains readable at the intended print width, and brief events keep their measured extent. Painstaking attention to timing and alignment prevents aesthetic convenience from changing the evidence.

Labels identify observations directly. A short semantic sentence gives each panel a purpose; exact measurements remain attached to their data. Master-level execution is expressed through restraint, clear hierarchy, and visible limits on interpretation.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FS.apply()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    a = load("msdwild_many", "01151", 435, 455)
    b = load("msdwild_many", "02311", 0, 54)
    counts = frame_counts(b)
    # Design rationale is written before visual expression, alongside the figure.
    (OUT / (STEM + "_philosophy.md")).write_text(PHILOSOPHY)
    fig = plt.figure(figsize=(W/72, H/72), facecolor="white")
    event = return_panel(fig, a)
    failure_panel(fig, b, counts)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in labels) >= 8
    outside = [t.get_text() for t in labels if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
               or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    assert not outside, f"Text outside fixed canvas: {outside}"
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(OUT / (STEM + suffix), dpi=300, bbox_inches=None)
    plt.close(fig)
    metadata = dict(width_pt=W, height_pt=H, minimum_label_pt=8,
                    threshold=">0.5", frame_s=.01, latency_profile_s=1.04, fine_tuned_seed=1,
                    selection="Existing largest gain and largest loss: 41 MSDWild-many recordings with >=7 speakers; extremes, not typical",
                    return_event=event, failure_speaker_counts=counts, sources=provenance([a, b]),
                    coverage_definition="100 * frames(reference AND mapped prediction) / frames(reference); no false-positive penalty; not DER",
                    text_boundary_check="passed")
    (OUT / (STEM + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / (STEM + "_review.md")).write_text(review_text(counts))
    print(f"Saved one prototype: {W} × {H} pt, minimum labels 8 pt; frame/timing/bounds assertions passed.")
    for suffix in [".pdf", ".svg", ".png", "_metadata.json", "_review.md"]:
        print(OUT / (STEM + suffix))


if __name__ == "__main__":
    main()
