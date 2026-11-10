#!/usr/bin/env python3
"""Run a supplied JSON configuration, with optional thread/round overrides."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("vigor", "hnsw", "prohnsw", "wolverine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=(*ENGINES, "all"))
    parser.add_argument("dataset", choices=("sift", "gist", "laion", "dbpedia", "deep100m"))
    parser.add_argument("--workload", choices=("random_update", "localized_update"), default="random_update")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--queries", type=int)
    parser.add_argument("--cpus", help="Comma-separated logical CPU IDs, preferably distinct physical cores")
    parser.add_argument("--name", help="Output name; engine is appended when running all")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    engines = ENGINES if args.engine == "all" else (args.engine,)
    build = args.build_dir.resolve()
    for engine in engines:
        original = ROOT / "config" / args.workload / args.dataset / f"{engine}.json"
        config = json.loads(original.read_text())
        if args.rounds is not None:
            config["rounds"] = args.rounds
        if args.queries is not None:
            config["queries"] = args.queries
        if args.threads is not None:
            config["threads"] = config["gt_threads"] = args.threads
            config["build_threads"] = min(config["build_threads"], args.threads)
        if args.cpus:
            config["cpus"] = [int(cpu) for cpu in args.cpus.split(",")]
        if args.name:
            config["name"] = f"{args.name}_{engine}" if args.engine == "all" else args.name
        elif args.rounds is not None:
            config["name"] += f"_r{args.rounds}"
        folder = build / "configs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{engine}_{args.workload}_{args.dataset}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        command = [str(build / f"benchmark_{engine}"), str(path)]
        if args.check:
            command.append("--check")
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
