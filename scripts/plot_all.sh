#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

found=0
for workload in random_update localized_update; do
    available=0
    for metric in exp_results/{vigor,hnsw,prohnsw,wolverine}_"${workload}"_{sift,gist,dbpedia,laion,deep100m}/metric.csv; do
        if [[ -f "$metric" ]]; then
            available=1
            break
        fi
    done
    if [[ "$available" == 1 ]]; then
        if [[ "$workload" == random_update ]]; then
            python3 scripts/plots/fig_for_random.py
        else
            python3 scripts/plots/fig_for_localized.py
        fi
        found=1
    fi
done
if [[ "$found" == 0 ]]; then
    echo 'No standard-named experiment CSVs found. Run scripts/run_experiment.py first, or pass explicit paths to a plot script.' >&2
    exit 1
fi
