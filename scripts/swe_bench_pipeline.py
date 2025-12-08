#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from editor.ast_parser import ProjectParser
from editor.code_tree import CodeSemanticTree
from agent.recap import RecapSWE
from llm.llm import create_prompt, create_llm

SWE_BENCH_DATASETS = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
}

SB_CLI_SUBSET_MAP = {
    "verified": "swe-bench_verified",
    "full": "swe-bench",
}

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "swe_bench"

logger = logging.getLogger("swe-bench-pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_instances(
    *,
    subset: str,
    split: str,
    dataset_path: Path | None,
    limit: int | None,
    filter_regex: str,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:

    """Load SWE-bench instances either from HuggingFace or a local JSON file."""
    if dataset_path is not None:
        logger.info("Loading SWE-bench instances from %s", dataset_path)
        suffix = dataset_path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "pandas and pyarrow are required to load parquet SWE-bench metadata"
                ) from exc
            raw = pd.read_parquet(dataset_path).to_dict(orient="records")
        elif suffix == ".jsonl":
            raw = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        else:
            raw = json.loads(dataset_path.read_text())
    else:
        dataset_name = SWE_BENCH_DATASETS.get(subset)
        if dataset_name is None:
            raise ValueError(f"Unsupported subset: {subset}")
        logger.info("Downloading SWE-bench dataset %s (%s)", dataset_name, split)
        from datasets import load_dataset

        raw = load_dataset(dataset_name, split=split)
    instances = list(raw)

    logger.info("Loaded %d instances before filtering", len(instances))
    pattern = re.compile(filter_regex)
    instances = [inst for inst in instances if pattern.match(inst["instance_id"])]
    logger.info("Filtered down to %d instances using pattern %r", len(instances), filter_regex)

    if shuffle:
        random.Random(seed).shuffle(instances)
        logger.info("Shuffled instances with seed %d", seed)

    if limit is not None:
        instances = instances[:limit]
        logger.info("Truncated to %d instances", len(instances))
    return instances


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logger.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=cwd, check=True)


def _write_instance_metadata(instances: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(instances, indent=2))
    logger.info("Wrote metadata for %d instances to %s", len(instances), destination)


def _clone_single_instance(instance: dict[str, Any], repo_dir: Path) -> bool:
    repo_url = f"https://github.com/{instance['repo']}.git"
    base_commit = instance["base_commit"]
    logger.info("Preparing %s (%s) at %s", instance["instance_id"], repo_url, repo_dir)
    try:
        if repo_dir.exists():
            _run(["git", "fetch", "origin"], cwd=repo_dir)
        else:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", repo_url, str(repo_dir)])
        _run(["git", "fetch", "origin", base_commit], cwd=repo_dir)
        _run(["git", "checkout", "--force", base_commit], cwd=repo_dir)
        _run(["git", "reset", "--hard", base_commit], cwd=repo_dir)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to prepare %s: %s", instance["instance_id"], exc)
        return False


def clone_instances(instances: list[dict[str, Any]], repos_dir: Path) -> None:
    repos_dir.mkdir(parents=True, exist_ok=True)
    for inst in instances:
        repo_dir = repos_dir / inst["instance_id"]
        _clone_single_instance(inst, repo_dir)


def build_tree_for_repo(repo_dir: Path, output_file: Path, summarize: bool, include_source: bool) -> None:
    logger.info("Building semantic tree for %s", repo_dir)
    parser = ProjectParser(str(repo_dir))
    parser.build_index()
    tree = CodeSemanticTree(parser, project_name=parser.root.name)
    tree.build_from_parser_with_folder(parser)
    if summarize:
        try:
            tree.generate_summary()
        except Exception as exc:
            logger.warning("Summary generation failed for %s: %s", repo_dir, exc)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tree.save_json(str(output_file), include_source=include_source)
    logger.info("Saved tree to %s", output_file)


def build_trees(
    instances: list[dict[str, Any]],
    repos_dir: Path,
    trees_dir: Path,
    summarize: bool,
    include_source: bool,
    skip_existing: bool,
    auto_clone: bool,
    cleanup_repos: bool,
) -> None:
    for inst in instances:
        repo_dir = repos_dir / inst["instance_id"]
        tree_file = trees_dir / f"{inst['instance_id']}.json"

        def cleanup_repo() -> None:
            if cleanup_repos and repo_dir.exists():
                logger.info("Removing repo clone for %s at %s", inst["instance_id"], repo_dir)
                shutil.rmtree(repo_dir, ignore_errors=True)

        if skip_existing and tree_file.exists():
            logger.info("Tree already exists for %s, skipping", inst["instance_id"])
            cleanup_repo()
            continue

        if not repo_dir.exists():
            if not auto_clone:
                logger.warning(
                    "Repository for %s not found at %s; rerun clone step or enable --auto-clone",
                    inst["instance_id"],
                    repo_dir,
                )
                continue
            if not _clone_single_instance(inst, repo_dir):
                continue
        try:
            build_tree_for_repo(repo_dir, tree_file, summarize, include_source)
        except Exception as exc:
            logger.error("Failed to build tree for %s: %s", inst["instance_id"], exc)
        else:
            cleanup_repo()


def create_prediction_file(instance: dict[str, Any], patch: str, run_name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pred_path = directory / f"{instance['instance_id']}.pred"
    datum = {
        "model_name_or_path": run_name,
        "instance_id": instance["instance_id"],
        "model_patch": patch,
    }
    pred_path.write_text(json.dumps(datum, indent=2))
    logger.debug("Wrote prediction %s", pred_path)
    return pred_path


def merge_predictions(prediction_dirs: list[Path], output: Path) -> Path:
    merged: dict[str, Any] = {}
    for directory in prediction_dirs:
        for pred in directory.rglob("*.pred"):
            data = json.loads(pred.read_text())
            instance_id = data["instance_id"]
            if instance_id in merged:
                continue 
            merged[instance_id] = data
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2))
    logger.info("Merged %d predictions into %s", len(merged), output)
    return output


def run_sb_cli(
    *, subset: str, split: str, predictions_path: Path, output_dir: Path, run_id: str | None = None
) -> None:
    dataset = SB_CLI_SUBSET_MAP.get(subset, subset)
    cmd = [
        "sb-cli",
        "submit",
        dataset,
        split,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id or predictions_path.stem,
        "--output_dir",
        str(output_dir / "sb-cli-reports"),
    ]
    logger.info("Running sb-cli: %s", " ".join(cmd))
    _run(cmd)


def generate_patch_from_tree(instance: dict[str, Any], tree_file: Path) -> str:
    """Generate a patch using tree and the SWE-bench problem.
    Currently returns the ground-truth patch as a placeholder.
    """
    tree_data = None
    if tree_file.exists():
        try:
            tree_data = json.loads(tree_file.read_text())
        except Exception as exc:
            logger.warning("Failed to read tree file %s: %s", tree_file, exc)
    else:
        logger.warning("Tree file %s not found; falling back to dataset patch", tree_file)

    _ = tree_data  # placeholder 
    _problem_statement = instance.get("problem_statement", "")
    _ = _problem_statement
    return instance.get("patch") or ""

def generate_patch_from_context(
    instance: dict[str, Any], 
    tree_file: Path, 
    context_file: Path,
    code_model: str = "gpt-4o-mini",
    **llm_kwargs
) -> str:
    """Generate a patch using context and the SWE-bench problem.
    """
    context_data = None
    if context_file.exists():
        try:
            context_data = json.loads(context_file.read_text())
        except Exception as exc:
            logger.warning("Failed to read context file %s: %s", context_file, exc)
    
    # tree_data = None
    # if tree_file.exists():
    #     try:
    #         tree_data = json.loads(tree_file.read_text())
    #     except Exception as exc:
    #         logger.warning("Failed to read tree file %s: %s", tree_file, exc)
    
    prompt = create_prompt(instance, context_data)
    
    # Call LLM to generate patch
    try:
        llm = create_llm(code_model, **llm_kwargs)
        patch = llm.generate(prompt)
        logger.info("patch: %s", patch)
        return patch
    except Exception as exc:
        logger.error("Failed to generate patch using LLM: %s", exc)
        return ""


def evaluate(
    instances: list[dict[str, Any]],
    predictions_dir: Path,
    run_name: str,
    *,
    patches_dir: Path | None,
    tree_dir: Path,
    context_dir: Path,
    subset: str,
    split: str,
    run_sb: bool,
    output_dir: Path,
    code_model: str = "gpt-4o-mini",
    **llm_kwargs,
) -> Path:
    for inst in instances:
        if patches_dir:
            patch_file = patches_dir / f"{inst['instance_id']}.patch"
            if not patch_file.exists():
                logger.warning("Patch file %s not found; skipping %s", patch_file, inst["instance_id"])
                continue
            patch_text = patch_file.read_text()
        else:
            tree_file = tree_dir / f"{inst['instance_id']}.json"
            # patch_text = generate_patch_from_tree(inst, tree_file)
            context_file = context_dir / f"{inst['instance_id']}.json"
            patch_text = generate_patch_from_context(inst, tree_file, context_file, code_model, **llm_kwargs)
        pred_dir = predictions_dir / inst["instance_id"]
        create_prediction_file(inst, patch_text, run_name, pred_dir)

    merged = merge_predictions([predictions_dir], output_dir / "preds.json")
    if run_sb:
        run_sb_cli(subset=subset, split=split, predictions_path=merged, output_dir=output_dir, run_id=run_name)
    return merged


def add_instance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subset", choices=sorted(SWE_BENCH_DATASETS.keys()), default="verified")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--dataset-path", type=Path, help="Optional local JSON file with SWE-bench instances")
    parser.add_argument("--limit", type=int, help="Process only the first N instances after filtering")
    parser.add_argument("--filter", default=".*", help="Regex applied to instance_id")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    parser.add_argument("--seed", type=int, default=13, help="Seed used when shuffling")


def add_tree_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_DATA_DIR / "repos")
    parser.add_argument("--trees-dir", type=Path, default=DEFAULT_DATA_DIR / "trees")
    parser.add_argument("--summarize", action="store_true", help="Call LLM summarizer if configured", default=True)
    parser.add_argument("--include-source", action="store_true", help="Store source in JSON output", default=True)
    parser.add_argument("--skip-existing", action="store_true", help="Avoid regenerating trees that already exist", default="True")
    parser.add_argument(
        "--auto-clone",
        action="store_true",
        help="Automatically clone repositories that are missing before building trees", default=True,
    )
    parser.add_argument(
        "--cleanup-repos",
        action="store_true",
        help="Delete the repository folder after a tree has been written or found", default=True,
    )

def add_context_args(parser: argparse.ArgumentParser, *, include_tree_dir: bool = False) -> None:
    if include_tree_dir:
        parser.add_argument("--trees-dir", type=Path, default=DEFAULT_DATA_DIR / "trees")
        parser.add_argument("--skip-existing", action="store_true", help="Avoid regenerating context that already exists", default=True)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_DATA_DIR / "context")

def add_eval_args(parser: argparse.ArgumentParser, *, include_tree_dir: bool = True, include_context_dir: bool = True) -> None:
    if include_tree_dir:
        parser.add_argument("--trees-dir", type=Path, default=DEFAULT_DATA_DIR / "trees")
    if include_context_dir:
        parser.add_argument("--context-dir", type=Path, default=DEFAULT_DATA_DIR / "context")
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_DATA_DIR / "predictions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR / "runs")
    parser.add_argument("--run-name", default="scope-ground-truth")
    parser.add_argument("--run-sb-cli", action="store_true", help="Submit preds.json via sb-cli")
    parser.add_argument("--patches-dir", type=Path, help="Directory with <instance_id>.patch files")
    parser.add_argument(
        "--code-model",
        default="gpt-4o-mini",
        help="LLM model name to use for patch generation (e.g., 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Maximum tokens for LLM generation"
    )
    


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clone_parser = subparsers.add_parser("clone", help="Clone SWE-bench repositories")
    add_instance_args(clone_parser)
    clone_parser.add_argument(
        "--repos-dir", type=Path, default=DEFAULT_DATA_DIR / "repos", help="Directory to store cloned repositories"
    )
    clone_parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_DATA_DIR / "instances.json",
        help="Where to store the filtered instances metadata",
    )

    tree_parser = subparsers.add_parser("trees", help="Build semantic trees for cloned repositories")
    add_instance_args(tree_parser)
    add_tree_args(tree_parser)

    context_parser = subparsers.add_parser("context", help="Generate context for instances")
    add_instance_args(context_parser)
    add_context_args(context_parser, include_tree_dir=True)

    eval_parser = subparsers.add_parser("evaluate", help="Create predictions and optionally run sb-cli evaluation")
    add_instance_args(eval_parser)
    add_eval_args(eval_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run tree generation followed by evaluation")
    add_instance_args(pipeline_parser)
    add_tree_args(pipeline_parser)
    add_context_args(pipeline_parser)
    add_eval_args(pipeline_parser, include_tree_dir=False, include_context_dir=False)

    return parser


def run_tree_stage(args, instances) -> None:
    build_trees(
        instances,
        repos_dir=args.repos_dir,
        trees_dir=args.trees_dir,
        summarize=args.summarize,
        include_source=args.include_source,
        skip_existing=args.skip_existing,
        auto_clone=args.auto_clone,
        cleanup_repos=args.cleanup_repos,
    )


def generate_context(instances: list[dict[str, Any]], tree_dir: Path, context_dir: Path, skip_existing: bool) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)

    for inst in instances:
        tree_file = tree_dir / f"{inst['instance_id']}.json"
        context_file = context_dir / f"{inst['instance_id']}.json"
        
        if skip_existing and context_file.exists():
            logger.info("Context already exists for %s, skipping", inst["instance_id"])
            continue

        tree = CodeSemanticTree(tree_path=tree_file)
        task_name =tree.nodes["root"].name
        task_description = inst.get("problem_statement", "")
        context_generator = RecapSWE(task_name=task_name, task_description=task_description, fewshot_example="", code_tree=tree)
        context, code_patch = context_generator.run()

        context_data = {
            "instance_id": inst["instance_id"],
            "context": context,
            "code_patch": code_patch,
        }
        context_file.write_text(json.dumps(context_data, indent=2))
        logger.info("Generated context for %s", inst["instance_id"])


def run_context_generation_stage(args, instances) -> None:
    generate_context(instances, args.trees_dir, args.context_dir, args.skip_existing)

def run_evaluation_stage(args, instances) -> Path:
    patches_dir = args.patches_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare LLM kwargs
    llm_kwargs = {}
    if hasattr(args, "llm_temperature"):
        llm_kwargs["temperature"] = args.llm_temperature
    if hasattr(args, "llm_max_tokens") and args.llm_max_tokens:
        llm_kwargs["max_tokens"] = args.llm_max_tokens
    
    return evaluate(
        instances,
        predictions_dir=args.predictions_dir,
        run_name=args.run_name,
        patches_dir=patches_dir,
        tree_dir=args.trees_dir,
        context_dir=args.context_dir,
        subset=args.subset,
        split=args.split,
        run_sb=args.run_sb_cli,
        output_dir=args.output_dir,
        code_model=getattr(args, "code_model", "gpt-4o-mini"),
        **llm_kwargs,
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = get_parser()
    args = parser.parse_args(argv)

    instances = _load_instances(
        subset=args.subset,
        split=args.split,
        dataset_path=args.dataset_path,
        limit=args.limit,
        filter_regex=args.filter,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if args.command == "clone":
        clone_instances(instances, args.repos_dir)
        _write_instance_metadata(instances, args.metadata_path)
    elif args.command == "trees":
        run_tree_stage(args, instances)
    elif args.command == "context":
        run_context_generation_stage(args, instances)
    elif args.command == "evaluate":
        merged = run_evaluation_stage(args, instances)
        logger.info("Predictions written to %s", merged)
    elif args.command == "pipeline":
        run_tree_stage(args, instances)
        run_context_generation_stage(args, instances)
        merged = run_evaluation_stage(args, instances)
        logger.info("Predictions written to %s", merged)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
