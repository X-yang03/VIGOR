#!/usr/bin/env python3
"""Download the paper datasets into datasets/, preserving their original row order."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAION = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-1m-no-filters.tgz"
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DEEP = HUB + "/datasets/suchun/deep1b-mirror/resolve/main/"


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_name(path.name + ".part")
    subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                    "--connect-timeout", "20", "--continue-at", "-",
                    url, "--output", str(partial)], check=True)
    partial.replace(path)


def write_fbin(path, vectors):
    if path.exists():
        return
    rows, dimension = vectors.shape
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as out:
        out.write(struct.pack("<II", rows, dimension))
        for start in range(0, rows, 8192):
            np.asarray(vectors[start:start + 8192], dtype="<f4").tofile(out)
    partial.replace(path)


def texmex(dataset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"{dataset}_base.fvecs", f"{dataset}_query.fvecs"]
    if all((directory / name).exists() for name in names):
        return
    if dataset == "sift":
        for name in names:
            fetch(HUB + "/datasets/qbo-odp/sift1m/resolve/main/" + name, directory / name)
        return
    archive = directory / f"{dataset}.tar.gz"
    fetch(HUB + "/datasets/fzliu/gist1m/resolve/main/gist.tar.gz", archive)
    with tarfile.open(archive) as source:
        for name in names:
            if (directory / name).exists():
                continue
            member = next(m for m in source.getmembers() if Path(m.name).name == name)
            partial = directory / (name + ".part")
            with source.extractfile(member) as inp, partial.open("wb") as out:
                shutil.copyfileobj(inp, out)
            partial.replace(directory / name)


def laion(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / name).exists() for name in ["laion_base.fbin", "laion_query.fbin"]):
        return
    archive = directory / "laion-1m-no-filters.tgz"
    fetch(LAION, archive)
    source = directory / "source"
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as data:
        for name in ["vectors.npy", "tests.jsonl"]:
            if (source / name).exists():
                continue
            member = next(m for m in data.getmembers() if Path(m.name).name == name)
            partial = source / (name + ".part")
            with data.extractfile(member) as inp, partial.open("wb") as out:
                shutil.copyfileobj(inp, out)
            partial.replace(source / name)
    vectors = np.load(source / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != 512:
        raise ValueError("Expected 512-dimensional LAION vectors")
    write_fbin(directory / "laion_base.fbin", vectors)
    with (source / "tests.jsonl").open() as inp:
        queries = np.asarray([json.loads(line)["query"] for line in inp if line.strip()], dtype="<f4")
    write_fbin(directory / "laion_query.fbin", queries)


def export_fvecs(path, data, begin, end):
    if path.exists():
        return
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as out:
        for first in range(begin, end, 8192):
            block = np.asarray(data[first:min(first + 8192, end)]["openai"], dtype="<f4")
            if block.shape[1] != 1536:
                raise ValueError("Expected 1536-dimensional DBpedia vectors")
            records = np.empty((len(block), 1537), dtype="<f4")
            records[:, 0] = np.asarray([1536], dtype="<i4").view("<f4")[0]
            records[:, 1:] = block
            records.tofile(out)
    partial.replace(path)


def dbpedia(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / name).exists() for name in ["dbpedia_all_base.fvecs", "dbpedia_all_query.fvecs"]):
        return
    from datasets import load_dataset
    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if len(data) != 1_000_000:
        raise ValueError("Expected the DBpedia 1M release")
    export_fvecs(directory / "dbpedia_all_base.fvecs", data, 0, 990_000)
    export_fvecs(directory / "dbpedia_all_query.fvecs", data, 990_000, 1_000_000)


def download_prefix(url, path, rows=200_000_000, dimension=96):
    """Resume a bounded prefix; require byte ranges to avoid downloading all 1B vectors."""
    import requests
    length = 8 + rows * dimension * 4
    if path.exists():
        return
    partial = path.with_name(path.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start < length:
        with requests.get(url, headers={"Range": f"bytes={start}-{length - 1}"},
                          stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{length - 1}/"
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError("The DEEP server did not honor the requested byte range")
            with partial.open("ab") as out:
                for block in response.iter_content(8 << 20):
                    out.write(block)
    if partial.stat().st_size != length:
        raise RuntimeError("Incomplete DEEP prefix; rerun to resume")
    with partial.open("r+b") as out:
        out.write(struct.pack("<II", rows, dimension))
    partial.replace(path)


def deep(directory):
    directory.mkdir(parents=True, exist_ok=True)
    download_prefix(DEEP + "base.1B.fbin", directory / "base.200M.fbin")
    fetch(DEEP + "query.public.10K.fbin", directory / "deep_query.fbin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("sift", "gist", "laion", "dbpedia", "deep100m"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if args.dataset in ("sift", "gist"):
        texmex(args.dataset, data / (args.dataset + "1m"))
    elif args.dataset == "laion":
        laion(data / "laion1m")
    elif args.dataset == "dbpedia":
        dbpedia(data / "dbpedia_1536d")
    else:
        deep(data / "deep200m")
    print("Ready:", args.dataset, "in", data)


if __name__ == "__main__":
    main()
