"""Shared Arial typography and headless PDF output for the paper figures."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "exp_results"
OUTPUT = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_COLOR = "#30353B"
TICK_COLOR = "#454B52"


def apply_paper_style(body=7.5, tick=7, legend=8.5, label=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": body, "font.weight": "normal",
        "axes.labelsize": body if label is None else label,
        "axes.labelweight": "normal", "axes.labelcolor": TEXT_COLOR,
        "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlecolor": TEXT_COLOR,
        "text.color": TEXT_COLOR, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "xtick.color": TICK_COLOR, "ytick.color": TICK_COLOR,
        "legend.fontsize": legend, "legend.labelcolor": TEXT_COLOR,
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#8D969E", "axes.linewidth": .55,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 150, "savefig.dpi": 300,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    })

