# Main-experiment configurations

40 JSON files cover four in-place methods × five datasets × two workloads:

```text
config/{random_update,localized_update}/{sift,gist,laion,dbpedia,deep100m}/
    vigor.json
    hnsw.json
    prohnsw.json
    wolverine.json
```

Pass one file directly to its matching executable; see [run/README.md](../run/README.md).
Data/cache paths inside JSON are relative to the repository root. Results are
always saved to `exp_results/{name}/metric.csv`. Each supplied JSON sets `name`
to `engine_workload_dataset`, e.g. `vigor_random_update_sift`. Passing the JSON
alone therefore uses that fixed directory. CLI `--name NAME` overrides it;
the parameter/timestamp fallback applies only when both names are absent.
JSON contains no output path. Defaults are eight
update callers, 100 rounds, 1% replacement per round, 1,000 queries, Recall@10,
M=16, efConstruction=200, efSearch=100 and seed=100.
The JSON files omit `seed`; the driver defaults to 100.

| Dataset | Distance | Initial active set | Batch | Physical update-pool offset | Index capacity |
| --- | --- | ---: | ---: | ---: | ---: |
| SIFT | L2 | 500,000 | 5,000 | 500,000 | 1,000,000 |
| GIST | L2 | 500,000 | 5,000 | 500,000 | 1,000,000 |
| LAION | Cosine | 500,000 | 5,000 | 500,000 | 1,000,000 |
| DBpedia | Cosine | 495,000 | 4,950 | 495,000 | 990,000 |
| Deep100M | L2 | 50,000,000 | 500,000 | **100,000,000** | 50,500,000 |

DBpedia reserves 10K vectors for its query file. Deep100M consumes the existing
Deep200M file: initial rows [0,50M), update pool [100M,150M). It does **not** use
rows [50M,100M) for updates.

`build_threads` applies only on an initial-cache miss: one for the four small
datasets and VIGOR Deep100M; eight for the native Deep100M cache. `threads` is
the update budget; `gt_threads` is an independent, sequential-phase GT budget.
Configurations omit host-specific CPU IDs; use `--cpus` in
`scripts/run_experiment.py` to select physical cores, or add a `cpus` JSON list. No experiment is started by configuring
or compiling the benchmarks.
