"""Paper style (constants measured from overleaf_repo/figures/*.pdf) and pt-based layout helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from analysis import config  # noqa: E402

INK = "#231f20"
Q_COLORS = {"high": "#ed2124", "mid": "#7f8080", "low": "#3953a4"}
Q_LEGEND = {"high": ">0.67", "mid": "", "low": "<0.33"}
RPE_COLORS = {">0.67": "#4fc0ac", "0.33": "#35a0ab", "0.0": "#367ca6",
              "-0.33": "#3c579a", "-0.67": "#3d366c", "-1.0": "#2b1c35"}
RPE_ORDER = [">0.67", "0.33", "0.0", "-0.33", "-0.67", "-1.0"]
HIST_NEG = ["#376cac", "#6083b0", "#8198b8", "#a2b0c6", "#c0c7d3", "#e0e2e8"]  # -6 .. -1
HIST_POS = ["#ecdcdb", "#dbbdbc", "#cc9f9d", "#be8280", "#ac6565", "#9a474a"]  # 1 .. 6
SCATTER = "#1d78b5"
FIT_NEG, FIT_POS = "#3953a4", "#ed2124"
CS_COLORS = {"CS1": "#f8961d", "CS2": "#5d53a3", "CS3": "#b9539f", "CS4": "#6d6e71"}
CS_LABEL = {"CS1": "10%", "CS2": "50%", "CS3": "90%", "CS4": "Airpuff"}
BASELINE_BOX = "#e6e7e8"
EVENT = {"reward": "#58c4e6", "right": "#a5a8d6", "left": "#f4a3a5", "gocue": "#b3b3b3", "ignored": "#1a9a3f"}
PROB_R, PROB_L = "#f4a3a5", "#b8bde0"
DA_COLOR, NE_COLOR = "#1d78b5", "#138a3e"
TRACE_GREEN = "#138a3e"
REGIME_COLORS = {"value_dominant": "#9a474a", "mixed": "#8e5ea2", "rpe_dominant": "#35a0ab", "weak": "#bcbec0"}
REGIME_LABEL = {"value_dominant": "Value", "mixed": "Mixed", "rpe_dominant": "RPE", "weak": "Weak"}
ANIMAL_CMAP = "tab10"
BOX_COLORS = {"task_forage": "#7fd1a6", "task_pav": "#68a283", "sys_da": "#7dc8ea", "sys_ne": "#6fa8d6"}

PAGE_W = 595.276
PT = 1 / 72.0


def setup() -> None:
    for f in fm.findSystemFonts():
        if "Arimo" in f:
            fm.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "Arimo", "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 1.1, "lines.solid_capstyle": "butt", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arimo", "mathtext.it": "Arimo:italic",
        "mathtext.bf": "Arimo:bold", "savefig.dpi": 300,
    })


@dataclass
class Page:
    """A figure laid out in pt from the top-left corner, like the Illustrator originals."""
    height: float
    width: float = PAGE_W
    snippets: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __post_init__(self):
        setup()
        self.fig = plt.figure(figsize=(self.width * PT, self.height * PT))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.width, 1 - (y + h) / self.height, w / self.width, h / self.height], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        self.fig.text(x / self.width, 1 - y / self.height, s, **kw)

    def letter(self, x, y, letter, title=None, size=12, title_size=8.5):
        self.text(x, y, letter, fontsize=size, fontweight="bold")
        if title:
            self.text(x + 13, y + 2.5, title, fontsize=title_size)

    def snippet(self, source: str, clip: tuple, dest: tuple):
        """Stamp region `clip` (pt, top-left origin) of overleaf figure `source` into `dest` (x, y, w, h)."""
        self.snippets.append((source, clip, dest))

    def save(self, name: str) -> Path:
        import pymupdf
        config.ensure_dirs()
        pdf = config.FIGURES / f"{name}.pdf"
        tmp = config.FIGURES / f".{name}_raw.pdf"
        self.fig.savefig(tmp)
        plt.close(self.fig)
        doc = pymupdf.open(tmp)
        page = doc[0]
        for source, clip, (x, y, w, h) in self.snippets:
            src = pymupdf.open(config.SNIPPET_SOURCE / source)
            page.show_pdf_page(pymupdf.Rect(x, y, x + w, y + h), src, 0,
                               clip=pymupdf.Rect(clip[0], clip[1], clip[2], clip[3]))
        doc.save(pdf, garbage=3, deflate=True)
        doc.close()
        tmp.unlink()
        png = pymupdf.open(pdf)[0].get_pixmap(dpi=200)
        png.save(config.FIGURES / f"{name}.png")
        return pdf


def trace(ax, t, mean, sem=None, color=INK, ls="-", lw=1.1, alpha=0.25, label=None, zorder=2):
    if sem is not None and np.any(np.isfinite(sem)):
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(t, mean, color=color, ls=ls, lw=lw, label=label, zorder=zorder)


def baseline_box(ax, lo=-1.0, hi=0.0, label="baseline", y_frac=1.0):
    ax.axvspan(lo, hi, color=BASELINE_BOX, lw=0, zorder=0)
    if label:
        ax.text(lo + 0.05, y_frac, label, transform=ax.get_xaxis_transform(), va="top", fontsize=5.5)


def window_box(ax, lo, hi, label):
    """Grey tab above the axis marking an analysis window, as in the original Fig 2/4 panels."""
    ax.axvspan(lo, hi, ymin=0.92, ymax=1.0, color=BASELINE_BOX, lw=0, zorder=0)
    ax.text(lo, 1.0, label, transform=ax.get_xaxis_transform(), va="bottom", fontsize=5.5)


def event_line(ax, x=0.0):
    ax.axvline(x, color="#bcbec0", lw=2.2, zorder=0, alpha=0.9)


def time_axis(ax, lo=-1, hi=4, label="Time - choice (s)"):
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(lo, hi + 1))
    ax.set_xlabel(label)


def scalebar(ax, x, y, length, label, horizontal=True, lw=1.0, fontsize=6):
    tr = ax.transData
    if horizontal:
        ax.plot([x, x + length], [y, y], color=INK, lw=lw, transform=tr, clip_on=False)
        ax.text(x + length / 2, y, label, ha="center", va="top", fontsize=fontsize, transform=tr)
    else:
        ax.plot([x, x], [y, y + length], color=INK, lw=lw, transform=tr, clip_on=False)
        ax.text(x, y + length / 2, " " + label, ha="left", va="center", fontsize=fontsize, transform=tr)


def nice_bar(span: float) -> float:
    """Round scale-bar length close to half of `span`."""
    target = span / 2
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if v >= target * 0.6:
            return float(v)
    return 200.0


def legend_q(ax, x0=0.62, y0=0.97, w=0.36, fontsize=4.2):
    """Header 'R+ R- Qch' and one row per Q tertile (solid R+, dashed R-), drawn in axes coordinates."""
    from matplotlib.patches import Rectangle
    tr = ax.transAxes
    row_h = 0.075
    ax.add_patch(Rectangle((x0, y0 - 4 * row_h - 0.02), w, 4 * row_h + 0.02, transform=tr, fill=False,
                           lw=0.4, ec="#d1d3d4", zorder=5))
    xs = [x0 + 0.03, x0 + 0.13, x0 + 0.24]
    for x, s in zip(xs, ["R+", "R-", "Qch"]):
        ax.text(x + 0.04, y0 - 0.01, s, transform=tr, fontsize=fontsize, ha="center", va="top", zorder=6)
    for i, q in enumerate(("high", "mid", "low")):
        y = y0 - (i + 1.45) * row_h
        ax.plot([xs[0], xs[0] + 0.08], [y, y], transform=tr, color=Q_COLORS[q], lw=1, zorder=6)
        ax.plot([xs[1], xs[1] + 0.08], [y, y], transform=tr, color=Q_COLORS[q], lw=1, ls=(0, (2, 1)), zorder=6)
        ax.text(xs[2] + 0.01, y, Q_LEGEND[q], transform=tr, fontsize=fontsize - 0.5, va="center", zorder=6)


def legend_rpe(ax, loc="upper right", fontsize=4.5):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RPE_COLORS[k], lw=1.2) for k in RPE_ORDER]
    leg = ax.legend(h, RPE_ORDER, loc=loc, fontsize=fontsize, title="RPE", title_fontsize=fontsize,
                    frameon=True, handlelength=1.4, borderpad=0.3, labelspacing=0.15)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#d1d3d4")
    return leg


def stat_text(ax, s, x=0.02, y=0.98, **kw):
    kw.setdefault("fontsize", 5)
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", **kw)


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def fmt_ci(d, digits=2):
    return f"{d['mean']:.{digits}f} [{d['ci_lo']:.{digits}f}, {d['ci_hi']:.{digits}f}]"


def animal_colors(animals):
    cmap = plt.get_cmap(ANIMAL_CMAP)
    return {a: cmap(i % 10) for i, a in enumerate(sorted(animals))}
