# Benchmark drivers

Build from the artifact root using its CMakeLists.txt:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/benchmark_vigor config/random_update/sift/vigor.json
```

The four executables are benchmark_vigor, benchmark_hnsw, benchmark_prohnsw,
and benchmark_wolverine. Each accepts `CONFIG.json [--name NAME] [--check]`.
The wrapper `scripts/run_experiment.py` also supports thread/round overrides.
Without baseline sources, only benchmark_vigor is built. Fetch refs and rerun
CMake configuration to add the native baseline targets; all need hnswlib for
their common initial-index constructor.

Source repositories are pinned under refs/. PRO-HNSW's src/hnswalg.h is overlaid
on the pinned hnswlib headers in the build directory. Wolverine keeps upstream
batch repair with delete model 4 and theta_in=32; its total thread budget includes
the caller. PRO-HNSW runs ROE each round and RDN/RUE every ten rounds.

Update timers include deletion/repair/insertion and worker joins, excluding I/O,
GT and query evaluation. VIGOR uses target-start repair search, ef_r=50, c=3,
serial one-shot prune, witness expansion, and no search witness learning.
All methods leave expand/checked/stale_hit/stale_ratio/dist_comp empty in new
CSVs. The shared 18-column CSV records recall and update timing.

Only metric.csv is saved under exp_results/{name}; an existing file is not
overwritten. Config/runtime copies live in build/configs. Initial graphs and GT
are cached by input paths and relevant parameters, without SHA fingerprints.
