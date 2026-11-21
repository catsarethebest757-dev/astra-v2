"""Command line: bootstrap, import, build, train, evaluate, benchmark, serve, similar, search, compare-index, self-test."""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

from .compare import compare_formats
from .config import DATA_DIR, BuildConfig
from .pipeline import bootstrap, build_features, evaluate_saved, load_engine, prepare, train_models
from .server import serve
from .util import relative_path


def _config(args) -> BuildConfig:
    return BuildConfig(encoder=args.encoder, adapter_epochs=args.adapter_epochs, ranker_epochs=args.ranker_epochs)


def _directory(args):
    return relative_path(args.dir)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=str(DATA_DIR))
    common.add_argument("--encoder", default="qwen3", choices=("qwen3", "jina5"))
    common.add_argument("--adapter-epochs", type=int, default=12)
    common.add_argument("--ranker-epochs", type=int, default=30)
    parser = argparse.ArgumentParser(prog="astranami", description="Astranami similarity ranking")
    commands = parser.add_subparsers(dest="command", required=True)

    boot = commands.add_parser("bootstrap", parents=[common], help="Download sources and build catalog.json")
    boot.add_argument("--tv-pages", type=int, default=1000)

    build = commands.add_parser("build", parents=[common], help="Encode aspects and build the feature index")
    build.add_argument("--force", action="store_true")

    train = commands.add_parser("train", parents=[common], help="Train the taste adapter and listwise ranker")
    train.add_argument("--no-refit", action="store_true")
    train.add_argument("--no-eval", action="store_true")

    commands.add_parser("evaluate", parents=[common], help="Re-score the saved held-out checkpoint")
    serve_cmd = commands.add_parser("serve", parents=[common], help="Serve the local discovery UI")
    serve_cmd.add_argument("--port", type=int, default=8765)
    commands.add_parser("prepare", parents=[common], help="Bootstrap, build, and train whatever is missing")
    prepare_cmd = commands.add_parser("prepare-serve", parents=[common], help="Prepare if needed, then serve")
    prepare_cmd.add_argument("--port", type=int, default=8765)

    similar = commands.add_parser("similar", parents=[common], help="Rank titles similar to one id")
    similar.add_argument("--id", required=True, action="append", help="Liked title id; repeat for several")
    similar.add_argument("--query", default="", help="Optional mood or plot description")
    similar.add_argument("--limit", type=int, default=12)
    similar.add_argument("--explain", action="store_true", help="Include the score breakdown for each result")

    search = commands.add_parser("search", parents=[common], help="Search titles")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=12)

    compare = commands.add_parser("compare-index", parents=[common],
                                  help="Measure what smaller vector formats would cost in neighbor quality")
    compare.add_argument("--samples", type=int, default=300)
    compare.add_argument("--k", type=int, default=20)

    importer = commands.add_parser("import", parents=[common],
                                   help="Build the catalog and index from exported titles and vectors")
    importer.add_argument("--titles", required=True, help="Title records as .jsonl or .jsonl.gz")
    importer.add_argument("--keys", required=True, help="JSON list of title ids, one per vector row")
    importer.add_argument("--vectors", required=True, help=".npy matrix of title vectors")
    importer.add_argument("--training", help="Folder with titles.jsonl and pairs.jsonl (families, recommendations)")

    bench = commands.add_parser("benchmark", parents=[common],
                                help="Score the engine and baselines against a judged file")
    bench.add_argument("file", help='JSON: {"group": [{"seed": id, "relevant": [ids]}], "pools": {"group": [ids]}}')
    bench.add_argument("--limit", type=int, help="Use only the first N seeds of each group")

    commands.add_parser("self-test", parents=[common], help="Run the unit tests")

    args = parser.parse_args(argv)
    directory = _directory(args)
    config = _config(args)
    if args.command == "bootstrap":
        bootstrap(directory, tv_pages=args.tv_pages)
    elif args.command == "build":
        build_features(directory, config, force=args.force)
    elif args.command == "train":
        train_models(directory, config, refit=not args.no_refit, run_eval=not args.no_eval)
    elif args.command == "evaluate":
        evaluate_saved(directory, config)
    elif args.command == "serve":
        serve(load_engine(directory), args.port)
    elif args.command in {"prepare", "prepare-serve"}:
        engine = prepare(directory, config)
        if args.command == "prepare-serve":
            serve(engine, args.port)
    elif args.command == "similar":
        payload = load_engine(directory).recommend(
            {"liked": args.id, "query": args.query, "limit": args.limit, "explain": args.explain})
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    elif args.command == "search":
        payload = load_engine(directory).lookup(args.query, limit=args.limit)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    elif args.command == "compare-index":
        payload = compare_formats(load_engine(directory).index, samples=args.samples, k=args.k)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    elif args.command == "import":
        from .importer import import_catalog
        import_catalog(directory, Path(args.titles), Path(args.keys), Path(args.vectors),
                       Path(args.training) if args.training else None)
    elif args.command == "benchmark":
        from .benchmark import benchmark_file
        engine = load_engine(directory)
        payload = benchmark_file(engine.index, Path(args.file), ranker=engine.ranker, limit=args.limit)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    elif args.command == "self-test":
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py", top_level_dir=".")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    else:
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
