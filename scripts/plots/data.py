"""Read benchmark outputs named engine_workload_dataset/metric.csv."""
from paper_style import RESULTS
import pandas as pd

DATASETS = ("sift", "gist", "dbpedia", "laion", "deep100m")
METHODS = ("vigor", "hnsw", "prohnsw", "wolverine")
BATCH_SIZE = {"sift": 5000, "gist": 5000, "laion": 5000,
              "dbpedia": 4950, "deep100m": 500000}


def trace_path(workload, dataset, method):
    return RESULTS / f"{method}_{workload}_{dataset}" / "metric.csv"


def read_trace(workload, dataset, method):
    return pd.read_csv(trace_path(workload, dataset, method)).sort_values("batch")

