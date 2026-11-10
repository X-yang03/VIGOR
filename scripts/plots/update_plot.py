"""Recall and update throughput from reproduced benchmark CSVs."""
from paper_style import OUTPUT, apply_paper_style, plt
from dataclasses import dataclass
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from data import DATASETS, METHODS, BATCH_SIZE, trace_path, read_trace

DATASET_TITLES = {"sift": "SIFT1M", "gist": "GIST1M", "dbpedia": "DBpedia1M",
                  "laion": "LAION1M", "deep100m": "Deep100M"}
METRIC_LABELS = {"recall": "Recall@10 (%)", "throughput": "Throughput\n(k updates/s)"}
VIGOR_METHODS = {"vigor"}

@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    linestyle: object
    marker: str
    linewidth: float = 0.95
    marker_phase: int = 0


STYLES = {
    "vigor": MethodStyle("VIGOR", "#E87524", "-", "o", 1.35, 0),
    "hnsw": MethodStyle("HNSW", "#4C78A8", (0, (4, 1.8)), "^", 0.95, 4),
    "prohnsw": MethodStyle("PRO-HNSW", "#619B87", (0, (2.5, 1.4)), "s", 0.95, 8),
    "wolverine": MethodStyle("Wolverine++", "#65758A", (0, (5, 1.5, 1, 1.5)), "v", 0.95, 16),
}


def values(df, metric):
    if metric == "throughput":
        df = df.loc[df["batch"] > 0]
    column = metric
    y = df[column].to_numpy(dtype=float)
    if metric == "recall":
        y = y * 100
    elif metric == "throughput":
        y = y / 1000
    return df["batch"].to_numpy(), y


def line_style(method):
    style = STYLES[method]
    highlighted = method in VIGOR_METHODS
    return {"label": style.label, "color": style.color, "linestyle": style.linestyle,
            "marker": style.marker, "linewidth": style.linewidth,
            "markersize": 2.2 if highlighted else 2.0, "markeredgewidth": 0.5,
            "markerfacecolor": style.color if highlighted else "white",
            "zorder": 6 if highlighted else 3,
            "solid_capstyle": "round", "dash_capstyle": "round"}


def marker_indices(method, length):
    """Stagger actual sample markers so nearby methods do not cover each other."""
    step = max(1, round(length / 5))
    offset = STYLES[method].marker_phase % step
    return list(range(offset, length, step))


def build_figure(path, data, datasets, methods, metrics, title="",
                 panel_aspect=2.0, width=8.4, legend_columns=4):
    """Use physical panel dimensions, independent of column and metric counts."""
    nrows, ncols = len(metrics), len(datasets)
    if width < 5:
        legend_columns = 2
    legend_count = len(methods)
    legend_rows = (legend_count + legend_columns - 1) // legend_columns
    # Margins/gaps are in inches, so changing the number of panels never turns
    # a landscape panel into a portrait panel. Keep the PDF's exact page size.
    left, right, col_gap, row_gap = 0.55, 0.10, 0.30, 0.30
    top = 0.26 + (0.20 if title else 0)
    bottom = 0.55 + 0.17 * (legend_rows - 1)
    panel_width = (width - left - right - col_gap * (ncols - 1)) / ncols
    panel_height = panel_width / panel_aspect
    height = top + bottom + nrows * panel_height + (nrows - 1) * row_gap
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex="col",
                             figsize=(width, height))
    fig.subplots_adjust(left=left / width, right=1 - right / width,
                        bottom=bottom / height, top=1 - top / height,
                        wspace=col_gap / panel_width, hspace=row_gap / panel_height)
    drawing_order = [m for m in methods if m not in VIGOR_METHODS] + [m for m in methods if m in VIGOR_METHODS]
    for col, dataset in enumerate(datasets):
        last_round = max(int(frame["batch"].iloc[-1]) for frame in data[dataset].values())
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            all_y = []
            for method in drawing_order:
                if method not in data[dataset]:
                    continue
                x, y = values(data[dataset][method], metric)
                if not np.isfinite(y).any():
                    continue
                all_y.extend(y[np.isfinite(y)])
                ax.plot(x, y, markevery=marker_indices(method, len(x)), **line_style(method))
            if row == 0:
                ax.set_title(f"({chr(97 + col)}) {DATASET_TITLES[dataset]}", fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], labelpad=6)
            ax.set_xlim(0, max(1, last_round))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4 if ncols >= 5 else 5, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, min_n_ticks=3,
                                                        steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
            if metric == "recall":
                low, high = min(all_y), max(all_y)
                pad = max(0.12, (high - low) * 0.10)
                ax.set_ylim(max(0, low - pad), min(100, high + pad))
            else:
                ax.set_ylim(0, max(all_y) * 1.08 if all_y else 1)
            ax.grid(color="#E0E5EA", linewidth=0.35, linestyle=(0, (2, 2)), zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(direction="out", pad=2)
    content_center = (left + width - right) / (2 * width)
    legend_methods = [methods[i] for col in range(min(legend_columns, legend_count))
                      for i in range(col, legend_count, legend_columns)]
    handles = [Line2D([], [], **line_style(m)) for m in legend_methods]
    labels = [STYLES[m].label for m in legend_methods]
    legend = fig.legend(handles, labels, loc="lower center",
                        bbox_to_anchor=(content_center, 0.02 / height),
                        ncol=min(legend_columns, legend_count), handlelength=2.4, handletextpad=0.45,
                        columnspacing=1.05, borderaxespad=0)
    for text, method in zip(legend.get_texts(), legend_methods):
        if method in VIGOR_METHODS:
            text.set_fontweight("bold")
            text.set_color(STYLES[method].color)
    if title:
        fig.suptitle(title, y=1 - 0.07 / height, fontsize=9, fontweight="bold")
    fig.supxlabel("Update round", fontsize=7.5, x=content_center,
                  y=(0.26 + 0.17 * (legend_rows - 1)) / height)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def input_paths(parser, paths, datasets, method, workload):
    """Accept a {dataset} template or paths in --datasets order."""
    if len(paths) not in (1, len(datasets)):
        parser.error("provide one path/template, or one path per --datasets entry")
    if (len(paths) == 1 and len(datasets) > 1 and "{dataset}" not in paths[0]
            and not Path(paths[0]).expanduser().is_dir()):
        parser.error("a single CSV requires one --datasets entry; use a template or path list for multiple datasets")
    resolved = {}
    for i, dataset in enumerate(datasets):
        path = Path(paths[0 if len(paths) == 1 else i].replace("{dataset}", dataset)).expanduser()
        if path.is_dir():
            # A parent directory may contain the standard engine_workload_dataset names.
            if len(paths) == 1 and len(datasets) > 1 and "{dataset}" not in paths[0]:
                path = path / f"{method}_{workload}_{dataset}"
            if path.is_dir():
                path = path / ("metric.csv" if (path / "metric.csv").is_file() else "metrics.csv")
        if not path.is_file():
            parser.error(f"{dataset}: input CSV not found: {path}")
        resolved[dataset] = path
    return resolved


def plot_updates(workload):
    parser = ArgumentParser(description=f"Plot {workload} from exp_results; write a PDF in figures/.")
    parser.add_argument("--output", type=Path, help="PDF destination")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        help="datasets in column/input-path order; default: available standard runs")
    for method in METHODS:
        parser.add_argument(f"--{method}", nargs="+", metavar="PATH",
                            help="CSV/result directories in dataset order, or a {dataset} path template")
    args = parser.parse_args()
    custom = any(getattr(args, method) is not None for method in METHODS)
    if custom and not args.datasets:
        parser.error("use --datasets when supplying explicit result paths")
    datasets = args.datasets or [d for d in DATASETS
                                if any(trace_path(workload, d, m).is_file() for m in METHODS)]
    if not datasets:
        parser.error("no experiment CSVs found; run scripts/run_experiment.py first")
    methods = [m for m in METHODS if getattr(args, m) is not None] if custom else list(METHODS)
    sources = {m: input_paths(parser, getattr(args, m), datasets, m, workload)
               for m in methods} if custom else {}
    data = {}
    for dataset in datasets:
        data[dataset] = {}
        for method in methods:
            if not custom and not trace_path(workload, dataset, method).is_file():
                continue
            frame = (pd.read_csv(sources[method][dataset]).sort_values("batch") if custom
                     else read_trace(workload, dataset, method))
            frame["throughput"] = BATCH_SIZE[dataset] / frame.update_total_s.where(frame.batch > 0)
            data[dataset][method] = frame
        if not data[dataset]:
            parser.error(f"{dataset}: no experiment CSVs found")
    methods = [m for m in methods if any(m in traces for traces in data.values())]
    apply_paper_style(legend=7.5)
    title = "Random Update" if workload == "random_update" else "Localized Update"
    destination = args.output if args.output else OUTPUT / f"{workload}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_figure(destination, data, datasets, methods, ["recall", "throughput"], title,
                 width=max(3.5, 1.68 * len(datasets)))
