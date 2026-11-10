# Scripts

- `datasets/download.py DATASET`: download/convert a dataset into `datasets/`.
- `run_experiment.py ENGINE DATASET`: run a JSON in `config/`; `--help` lists overrides.
- `plot_all.sh`: plot available default-named main experiments from `exp_results/`.
- `plots/fig_for_random.py` and `plots/fig_for_localized.py`: plot recall and update throughput.
- `clean_plot.sh`: remove the two generated main-experiment PDFs.

Entry points locate the repository relative to their own files. Explicit input
and output paths are relative to the caller. Plots read CSVs directly.

With default run names, dataset/method detection is automatic:

```bash
python3 scripts/run_experiment.py all sift
bash scripts/plot_all.sh
```

For a short run or custom names, specify the result paths:

```bash
python3 scripts/run_experiment.py vigor sift --rounds 20 --name sift_check
python3 scripts/plots/fig_for_random.py --datasets sift --vigor exp_results/sift_check --output figures/sift_check.pdf
```

`--hnsw`, `--prohnsw`, and `--wolverine` add methods in the same way.
For multiple datasets, pass paths in `--datasets` order or use a `{dataset}`
path template. Only explicitly supplied methods are drawn in this mode.
Default outputs are `figures/random_update.pdf` and `figures/localized_update.pdf`.
Arial is used when installed, with a sans-serif fallback.
Throughput uses the batch sizes in the supplied configurations.

Dataset sources: TexMex SIFT/GIST via Hugging Face mirrors (`qbo-odp/sift1m`,
`fzliu/gist1m`); Qdrant LAION no-filters; Hugging Face
`KShivendu/dbpedia-entities-openai-1M`; DEEP via `suchun/deep1b-mirror`.
Downloads support resuming and retain completed data files.
For Hugging Face files, `HF_ENDPOINT=https://hf-mirror.com` selects a mirror.
