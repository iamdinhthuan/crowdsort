"""Shared figure design system for the CrowdSort paper (two-column elsarticle 5p layout).

Tokens: palette #2563EB / #F59E0B / #10B981 / #8B5CF6 / #64748B (slate) / #F8FAFC (surface); one sans-serif
family (Arial, or the metric-compatible Liberation Sans); 7-8 pt labels, 9 pt bold panel labels; thin dark-grey
axes and light horizontal grid lines; widths equal to the LaTeX \\columnwidth (252 pt) and \\textwidth (522 pt),
so that figures are included without rescaling.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PT = 1 / 72.0
COL_W = 252 * PT  # inches
TEXT_W = 522 * PT

INK = "#1E293B"
AXIS = "#334155"
GRID = "#E2E8F0"
SLATE = "#64748B"
SURFACE = "#F8FAFC"
UNMATCHED = "#CBD5E1"

# one colour, marker and line style per system, identical in every figure
SYSTEMS = {
    "N3D": dict(color="#64748B", marker="o", ls="-", hatch=""),
    "FT-meet": dict(color="#94A3B8", marker="v", ls="-", hatch=""),
    "FT-clips": dict(color="#F59E0B", marker="s", ls="-", hatch=""),
    "SimConv-MSD": dict(color="#10B981", marker="^", ls="-", hatch=""),
    "FastMSS": dict(color="#8B5CF6", marker="D", ls="-", hatch=""),
    "CrowdSort-C": dict(color="#60A5FA", marker="P", ls="-", hatch=""),
    "CrowdSort": dict(color="#2563EB", marker="o", ls="-", hatch=""),
    "CrowdSort-P": dict(color="#1E3A8A", marker="d", ls="--", hatch="////"),
}

# categorical colours for reference speakers in timelines (colour-blind aware order)
SPEAKERS = ["#2563EB", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444", "#06B6D4", "#EC4899", "#84CC16",
            "#A16207", "#0F172A"]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Arimo", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "standard",
        "figure.dpi": 150,
    })


def panel_label(ax, text, x=-0.02, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom", color=INK)


def save(fig, out_pdf):
    fig.savefig(out_pdf)
    fig.savefig(out_pdf[:-4] + ".svg")
