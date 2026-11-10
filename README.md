# VIGOR artifact

Source code for **VIGOR: Efficient In-Place Updates for Navigable Graph Indexes via Reciprocal Coverage**.

## Getting Started
C++17, CMake >=3.16, OpenMP, Boost headers, Git, curl, and Python 3 are required.
On Ubuntu:

```bash
sudo apt-get install build-essential cmake libboost-dev git curl python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r scripts/requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

## Public Experiments
Download SIFT and check a short VIGOR run:

```bash
python3 scripts/datasets/download.py sift
python3 scripts/run_experiment.py vigor sift --rounds 20 --name sift_check
python3 scripts/plots/fig_for_random.py --datasets sift --vigor exp_results/sift_check --output figures/sift_check.pdf
```

Run all four methods with the supplied 100-round, eight-thread configurations:

```bash
bash refs/fork_baseline.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
python3 scripts/run_experiment.py all sift
python3 scripts/run_experiment.py all sift --workload localized_update
bash scripts/plot_all.sh
```

Methods: `vigor`, `hnsw`, `prohnsw`, `wolverine`, or `all`.
Datasets: `sift`,`gist`, `laion`, `dbpedia`, `deep100m`. Download each dataset before running it.

Use `--cpus 0,1,2,3,4,5,6,7` to select eight physical cores, or `--threads N`
to change the update and GT budgets. JSON files can also be passed directly:
`./build/benchmark_vigor config/random_update/sift/vigor.json`.
See [config/README.md](config/README.md) for dataset sizes and parameters.

Multithreaded updates may introduce slight run-to-run variation in recall; use `--threads 1` for more stable repeat measurements.

Results are saved as `exp_results/{name}/metric.csv`; existing files are not
overwritten. Use `--name` for repeated runs. Initial index/GT caches live under
`datasets/`. GT is shared across methods and run lengths.

`plot_all.sh` discovers available default-named runs in `exp_results/` and saves
recall/update-throughput PDFs in `figures/`.

VIGOR is distributed under [Apache-2.0](LICENSE). It includes code derived from
[hnswlib](https://github.com/nmslib/hnswlib), also licensed under Apache-2.0.
Baseline source repositories retain their original notices and licenses in `refs/`.
