#!/usr/bin/env python3
"""Orchestrate multi-seed BART/proxy ablation experiments and aggregate variance.

This script delegates to existing entrypoints (train_stage2.py, train_proxy.py,
run_stage2.py, run_proxy_pipeline.py, evaluate.py) with seed-specific output
directories, then aggregates evaluation_summary.json files to compute per-metric
mean and sample variance across seeds.

Default held-out split: /scratch/stages/hmr_stage2_dataset/test.jsonl

Subcommands:
  train-bart      Train one BART condition for one seed.
  train-seed      Train all conditions + proxy for one seed.
  train-proxy     Train proxy for one seed using that seed's full-condition BART.
  evaluate-seed   Run inference and evaluation for one seed.
  seed-pipeline   Full per-seed train + optional evaluate.
  aggregate       Aggregate per-seed evaluation_summary.json files.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_CONDITIONS = ["full", "target_only", "visual_only", "none"]
METRIC_KEYS = [
    "text_sta",
    "text_sta_delta",
    "sim",
    "clip",
]


def repo_default() -> Path:
    return Path(__file__).resolve().parents[1]


def as_cmd_text(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_logged(
    cmd: Sequence[object],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    """Execute cmd as a subprocess, teeing merged stdout/stderr to log_path.

    Raises RuntimeError on non-zero exit to propagate failures to the orchestrator.
    dry_run prints the command and returns without executing.
    """
    cmd_text = as_cmd_text(cmd)
    print(f"\n[command] {cmd_text}")
    print(f"[log]     {log_path}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {cmd_text}\n\n")
        log.flush()
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {cmd_text}")


def common_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.hf_cache:
        env["HF_HOME"] = str(args.hf_cache)
    return env


def dataset_dir(args: argparse.Namespace) -> Path:
    return args.dataset_dir or args.stages_root / "hmr_stage2_dataset"


def stage1_output_dir(args: argparse.Namespace) -> Path:
    return args.stage1_output_dir or args.stages_root / "hmr_stage1_output"


def validation_jsonl(args: argparse.Namespace) -> Path:
    return args.validation_jsonl or dataset_dir(args) / "test.jsonl"


def seed_stage_root(args: argparse.Namespace, seed: int) -> Path:
    return args.stages_root / args.experiment_name / f"seed_{seed}"


def seed_eval_root(args: argparse.Namespace, seed: int) -> Path:
    return args.eval_root / args.experiment_name / f"seed_{seed}"


def checkpoint_dir(args: argparse.Namespace, seed: int, condition: str) -> Path:
    return (
        seed_stage_root(args, seed)
        / f"hmr_stage2_{condition}{args.output_suffix}_checkpoint"
    )


def proxy_dir(args: argparse.Namespace, seed: int) -> Path:
    return seed_stage_root(args, seed) / f"hmr_proxy_checkpoint{args.output_suffix}"


def finetuned_eval_dir(args: argparse.Namespace, seed: int, condition: str) -> Path:
    return seed_eval_root(args, seed) / f"hmr_eval_stage2_{condition}{args.output_suffix}"


def bart_base_eval_dir(args: argparse.Namespace, seed: int, condition: str) -> Path:
    return seed_eval_root(args, seed) / f"hmr_eval_bart_base_{condition}{args.output_suffix}"


def proxy_eval_dir(args: argparse.Namespace, seed: int) -> Path:
    return seed_eval_root(args, seed) / f"hmr_eval_clip_proxy_bart_full{args.output_suffix}"


def detoxllm_eval_dir(args: argparse.Namespace, seed: int) -> Path:
    return seed_eval_root(args, seed) / f"hmr_eval_detoxllm{args.output_suffix}"


def results_dir(args: argparse.Namespace, seed: int) -> Path:
    return seed_eval_root(args, seed) / f"hmr_eval_results{args.output_suffix}"


def aggregate_dir(args: argparse.Namespace) -> Path:
    return args.eval_root / args.experiment_name / "aggregate"


def count_jsonl(path: Path) -> int:
    """Return the number of non-empty lines in a JSONL file, or -1 if absent."""
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def output_complete(path: Path, expected_count: int) -> bool:
    """Return True if path exists and contains exactly expected_count lines."""
    return expected_count > 0 and count_jsonl(path) == expected_count


def maybe_task_prefix_args(args: argparse.Namespace) -> List[str]:
    if getattr(args, "task_prefix", ""):
        return ["--task_prefix", args.task_prefix]
    return []


def maybe_debug_arg(args: argparse.Namespace) -> List[str]:
    return ["--debug"] if getattr(args, "debug", False) else []


def require_path(path: Path, label: str, dry_run: bool = False) -> None:
    if dry_run:
        return
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def train_bart(args: argparse.Namespace, seed: int, condition: str) -> None:
    out_dir = checkpoint_dir(args, seed, condition)
    history_path = out_dir / "training_history.json"
    config_path = out_dir / "config.json"
    if not args.force and history_path.exists() and config_path.exists():
        print(f"[skip] BART seed={seed} condition={condition}: {out_dir}")
        return

    require_path(dataset_dir(args), "Stage 2 dataset dir", args.dry_run or args.debug)
    require_path(stage1_output_dir(args), "Stage 1 output dir", args.dry_run or args.debug)

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "training" / "train_stage2.py",
        "--condition",
        condition,
        "--dataset_dir",
        dataset_dir(args),
        "--output_dir",
        out_dir,
        "--hf_cache",
        args.hf_cache,
        "--stage1_output_dir",
        stage1_output_dir(args),
        "--input_format",
        args.input_format,
        "--num_train_epochs",
        args.num_train_epochs,
        "--per_device_train_batch_size",
        args.per_device_train_batch_size,
        "--learning_rate",
        args.learning_rate,
        "--warmup_steps",
        args.warmup_steps,
        "--weight_decay",
        args.weight_decay,
        "--lora_r",
        args.lora_r,
        "--lora_alpha",
        args.lora_alpha,
        "--lora_dropout",
        args.lora_dropout,
        "--seed",
        seed,
    ]
    cmd.extend(["--base_model", args.base_model])
    cmd.extend(maybe_task_prefix_args(args))
    cmd.extend(maybe_debug_arg(args))

    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=seed_stage_root(args, seed) / "logs" / f"train_bart_{condition}.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )


def train_seed(args: argparse.Namespace, seed: int) -> None:
    for condition in args.conditions:
        train_bart(args, seed, condition)
    if not args.skip_proxy:
        train_proxy(args, seed)


def train_proxy(args: argparse.Namespace, seed: int) -> None:
    bart_dir = checkpoint_dir(args, seed, "full")
    out_dir = proxy_dir(args, seed)
    best_proxy = out_dir / "best_proxy.pt"
    history_path = out_dir / "training_history.json"
    if not args.force and best_proxy.exists() and history_path.exists():
        print(f"[skip] proxy seed={seed}: {out_dir}")
        return

    require_path(bart_dir, "Full-condition BART checkpoint", args.dry_run)
    require_path(dataset_dir(args), "Stage 2 dataset dir", args.dry_run or args.debug)
    require_path(stage1_output_dir(args), "Stage 1 output dir", args.dry_run or args.debug)

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "training" / "train_proxy.py",
        "--stage1_output_dir",
        stage1_output_dir(args),
        "--stage2_dataset_dir",
        dataset_dir(args),
        "--bart_checkpoint_dir",
        bart_dir,
        "--output_dir",
        out_dir,
        "--hf_cache",
        args.hf_cache,
        "--num_train_epochs",
        args.proxy_num_train_epochs,
        "--batch_size",
        args.proxy_batch_size,
        "--learning_rate",
        args.proxy_learning_rate,
        "--num_soft_tokens",
        args.proxy_num_soft_tokens,
        "--input_format",
        args.input_format,
        "--seed",
        seed,
    ]
    cmd.extend(maybe_task_prefix_args(args))
    cmd.extend(maybe_debug_arg(args))

    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=seed_stage_root(args, seed) / "logs" / "train_proxy.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )


def run_bart_inference(
    args: argparse.Namespace,
    *,
    seed: int,
    condition: str,
    checkpoint: str | Path,
    output_dir: Path,
    expected_count: int,
    inference_seed: int,
) -> None:
    output_file = output_dir / f"stage2_rewrites_{condition}.jsonl"
    if not args.force and output_complete(output_file, expected_count):
        print(f"[skip] inference {output_file} ({expected_count}/{expected_count})")
        return

    if isinstance(checkpoint, Path):
        require_path(checkpoint, f"BART checkpoint for {condition}", args.dry_run)

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "inference" / "run_stage2.py",
        "--condition",
        condition,
        "--checkpoint_dir",
        checkpoint,
        "--input_jsonl",
        validation_jsonl(args),
        "--output_dir",
        output_dir,
        "--hf_cache",
        args.hf_cache,
        "--input_format",
        args.input_format,
        "--batch_size",
        args.bart_eval_batch_size,
        "--num_beams",
        args.num_beams,
        "--max_length",
        args.max_length,
        "--seed",
        inference_seed,
    ]
    cmd.extend(maybe_task_prefix_args(args))
    cmd.extend(maybe_debug_arg(args))
    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=output_dir / "orchestrator_inference.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )


def run_proxy_inference(
    args: argparse.Namespace,
    *,
    seed: int,
    expected_count: int,
) -> Optional[Path]:
    out_dir = proxy_eval_dir(args, seed)
    output_file = out_dir / "stage2_rewrites_clip_proxy_bart_full.jsonl"
    if not args.force and output_complete(output_file, expected_count):
        print(f"[skip] proxy inference {output_file} ({expected_count}/{expected_count})")
        return out_dir

    bart_dir = checkpoint_dir(args, seed, "full")
    proxy_checkpoint = proxy_dir(args, seed) / "best_proxy.pt"
    require_path(bart_dir, "Full-condition BART checkpoint", args.dry_run)
    require_path(proxy_checkpoint, "Proxy checkpoint", args.dry_run)

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "inference" / "run_proxy_pipeline.py",
        "--input_jsonl",
        validation_jsonl(args),
        "--bart_checkpoint",
        bart_dir,
        "--proxy_checkpoint",
        proxy_checkpoint,
        "--output_dir",
        out_dir,
        "--hf_cache",
        args.hf_cache,
        "--text_prompt_format",
        args.proxy_text_prompt_format,
        "--batch_size",
        args.proxy_eval_batch_size,
        "--num_beams",
        args.num_beams,
        "--max_length",
        args.max_length,
        "--seed",
        seed,
    ]
    cmd.extend(maybe_debug_arg(args))
    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=out_dir / "orchestrator_proxy_inference.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )
    return out_dir


def run_detoxllm_if_requested(
    args: argparse.Namespace,
    *,
    seed: int,
    expected_count: int,
) -> Optional[Path]:
    if not args.include_detoxllm:
        return None
    out_dir = detoxllm_eval_dir(args, seed)
    output_file = out_dir / "detoxllm_rewrites.jsonl"
    if not args.force and output_complete(output_file, expected_count):
        print(f"[skip] DetoxLLM {output_file} ({expected_count}/{expected_count})")
        return out_dir

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "baselines" / "run_detoxllm_baseline.py",
        "--validation_jsonl",
        validation_jsonl(args),
        "--output_dir",
        out_dir,
        "--hf_cache",
        args.hf_cache,
        "--seed",
        args.baseline_seed,
    ]
    cmd.extend(maybe_debug_arg(args))
    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=out_dir / "orchestrator_detoxllm.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )
    return out_dir


def evaluate_seed(args: argparse.Namespace, seed: int) -> None:
    test_path = validation_jsonl(args)
    require_path(test_path, "Held-out test JSONL", args.dry_run)
    expected_count = count_jsonl(test_path)
    if expected_count <= 0 and not args.dry_run:
        raise RuntimeError(f"No examples found in held-out test JSONL: {test_path}")

    print(f"[eval] seed={seed} held-out set={test_path} n={expected_count}")

    bart_base_dirs: List[Path] = []
    if args.include_bart_base:
        for condition in args.conditions:
            out_dir = bart_base_eval_dir(args, seed, condition)
            run_bart_inference(
                args,
                seed=seed,
                condition=condition,
                checkpoint=args.base_model,
                output_dir=out_dir,
                expected_count=expected_count,
                inference_seed=args.baseline_seed,
            )
            bart_base_dirs.append(out_dir)

    finetuned_dirs: List[Path] = []
    for condition in args.conditions:
        out_dir = finetuned_eval_dir(args, seed, condition)
        run_bart_inference(
            args,
            seed=seed,
            condition=condition,
            checkpoint=checkpoint_dir(args, seed, condition),
            output_dir=out_dir,
            expected_count=expected_count,
            inference_seed=seed,
        )
        finetuned_dirs.append(out_dir)

    proxy_dirs: List[Path] = []
    if not args.skip_proxy:
        proxy_out = run_proxy_inference(args, seed=seed, expected_count=expected_count)
        if proxy_out is not None:
            proxy_dirs.append(proxy_out)

    detoxllm_dir = run_detoxllm_if_requested(args, seed=seed, expected_count=expected_count)

    cmd: List[object] = [
        sys.executable,
        args.repo_root / "evaluation" / "evaluate.py",
        "--validation_jsonl",
        test_path,
        "--output_dir",
        results_dir(args, seed),
        "--hf_cache",
        args.hf_cache,
        "--seed",
        seed,
    ]
    if bart_base_dirs:
        cmd.append("--bart_base_output_dirs")
        cmd.extend(bart_base_dirs)
    if finetuned_dirs:
        cmd.append("--bart_finetuned_output_dirs")
        cmd.extend(finetuned_dirs)
    if proxy_dirs:
        cmd.append("--proxy_output_dirs")
        cmd.extend(proxy_dirs)
    if detoxllm_dir is not None:
        cmd.extend(["--detoxllm_output_path", detoxllm_dir])
    if args.skip_clipscore:
        cmd.append("--skip_clipscore")
    cmd.extend(maybe_debug_arg(args))

    run_logged(
        cmd,
        cwd=args.repo_root,
        log_path=results_dir(args, seed) / "orchestrator_evaluate.log",
        env=common_env(args),
        dry_run=args.dry_run,
    )


def seed_pipeline(args: argparse.Namespace, seed: int) -> None:
    train_seed(args, seed)
    if args.run_eval_after_training:
        evaluate_seed(args, seed)


def read_summary(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def numeric_value(value: Any) -> Optional[float]:
    """Cast value to float for metric aggregation; returns None for booleans and non-numerics.

    Booleans are excluded explicitly because isinstance(True, int) is True in Python.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def aggregate(args: argparse.Namespace) -> None:
    per_seed_rows: List[Dict[str, Any]] = []
    values: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    missing: List[Path] = []

    for seed in args.seeds:
        summary_path = results_dir(args, seed) / "evaluation_summary.json"
        if not summary_path.exists():
            missing.append(summary_path)
            continue
        for row in read_summary(summary_path):
            system = str(row.get("system") or "unknown")
            per_seed_row = {"seed": seed, **row}
            per_seed_rows.append(per_seed_row)
            for metric in args.metrics:
                value = numeric_value(row.get(metric))
                if value is None:
                    continue
                values.setdefault(system, {}).setdefault(metric, []).append((seed, value))

    out_dir = aggregate_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, Any]] = []
    for system in sorted(values):
        for metric in sorted(values[system]):
            seed_values = values[system][metric]
            nums = [value for _, value in seed_values]
            row = {
                "system": system,
                "metric": metric,
                "n": len(nums),
                "expected_seeds": len(args.seeds),
                "mean": statistics.fmean(nums),
                "sample_variance": statistics.variance(nums) if len(nums) >= 2 else 0.0,
                "population_variance": statistics.pvariance(nums) if len(nums) >= 2 else 0.0,
                "sample_std": statistics.stdev(nums) if len(nums) >= 2 else 0.0,
                "min": min(nums),
                "max": max(nums),
                "values_by_seed": {str(seed): value for seed, value in seed_values},
            }
            aggregate_rows.append(row)

    payload = {
        "experiment_name": args.experiment_name,
        "seeds": args.seeds,
        "output_suffix": args.output_suffix,
        "summary_files_missing": [str(path) for path in missing],
        "metric_keys": args.metrics,
        "aggregates": aggregate_rows,
    }
    (out_dir / "metric_variance_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    aggregate_headers = [
        "system",
        "metric",
        "n",
        "expected_seeds",
        "mean",
        "sample_variance",
        "population_variance",
        "sample_std",
        "min",
        "max",
        "values_by_seed",
    ]
    with (out_dir / "metric_variance_summary.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(aggregate_headers) + "\n")
        for row in aggregate_rows:
            formatted = []
            for key in aggregate_headers:
                value = row.get(key)
                if isinstance(value, float):
                    formatted.append(f"{value:.6f}")
                elif isinstance(value, dict):
                    formatted.append(json.dumps(value, sort_keys=True))
                else:
                    formatted.append(str(value))
            f.write("\t".join(formatted) + "\n")

    per_seed_headers = ["seed", "system", "n", "valid_images", *args.metrics]
    with (out_dir / "per_seed_metrics.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(per_seed_headers) + "\n")
        for row in sorted(per_seed_rows, key=lambda x: (int(x["seed"]), str(x.get("system")))):
            formatted = []
            for key in per_seed_headers:
                value = row.get(key)
                if isinstance(value, float):
                    formatted.append(f"{value:.6f}")
                elif value is None:
                    formatted.append("")
                else:
                    formatted.append(str(value))
            f.write("\t".join(formatted) + "\n")

    print(f"Aggregate JSON: {out_dir / 'metric_variance_summary.json'}")
    print(f"Aggregate TSV:  {out_dir / 'metric_variance_summary.tsv'}")
    print(f"Per-seed TSV:   {out_dir / 'per_seed_metrics.tsv'}")
    if missing:
        print("[warn] Missing summaries:")
        for path in missing:
            print(f"  {path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo_root", type=Path, default=repo_default())
    parser.add_argument("--stages_root", type=Path, default=Path("/scratch/stages"))
    parser.add_argument("--eval_root", type=Path, default=Path("/scratch/eval_results"))
    parser.add_argument("--hf_cache", type=Path, default=Path("/scratch/hf_cache"))
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--stage1_output_dir", type=Path, default=None)
    parser.add_argument("--validation_jsonl", type=Path, default=None)
    parser.add_argument("--experiment_name", default="hmr_multiseed_explicit_detox")
    parser.add_argument("--output_suffix", default="_explicit_detox")
    parser.add_argument("--input_format", choices=["legacy", "explicit_detox"], default="explicit_detox")
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute even if expected outputs exist")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base_model", default="facebook/bart-large")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


def add_proxy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proxy_num_train_epochs", type=int, default=20)
    parser.add_argument("--proxy_batch_size", type=int, default=64)
    parser.add_argument("--proxy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--proxy_num_soft_tokens", type=int, default=16)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--bart_eval_batch_size", type=int, default=4)
    parser.add_argument("--proxy_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--proxy_text_prompt_format",
        choices=["none_legacy", "none_explicit_detox"],
        default="none_explicit_detox",
    )
    parser.add_argument("--include_bart_base", action="store_true")
    parser.add_argument("--include_detoxllm", action="store_true")
    parser.add_argument("--baseline_seed", type=int, default=42)
    parser.add_argument("--skip_proxy", action="store_true")
    parser.add_argument("--skip_clipscore", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-seed BART/proxy ablation orchestration and variance aggregation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("train-bart", help="Train one BART condition for one seed")
    add_common_args(p)
    add_train_args(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", choices=DEFAULT_CONDITIONS, required=True)
    p.set_defaults(func=lambda args: train_bart(args, args.seed, args.condition))

    p = subparsers.add_parser("train-seed", help="Train all BART conditions, then proxy, for one seed")
    add_common_args(p)
    add_train_args(p)
    add_proxy_args(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--skip_proxy", action="store_true")
    p.set_defaults(func=lambda args: train_seed(args, args.seed))

    p = subparsers.add_parser("train-proxy", help="Train proxy for one seed using that seed's full BART")
    add_common_args(p)
    add_proxy_args(p)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda args: train_proxy(args, args.seed))

    p = subparsers.add_parser("evaluate-seed", help="Run test inference and evaluation for one seed")
    add_common_args(p)
    add_eval_args(p)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda args: evaluate_seed(args, args.seed))

    p = subparsers.add_parser(
        "seed-pipeline",
        help="Train all conditions, train proxy, and optionally evaluate one seed",
    )
    add_common_args(p)
    add_train_args(p)
    add_proxy_args(p)
    add_eval_args(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_eval_after_training", action="store_true")
    p.set_defaults(func=lambda args: seed_pipeline(args, args.seed))

    p = subparsers.add_parser("aggregate", help="Aggregate per-seed evaluation summaries")
    add_common_args(p)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--metrics", nargs="+", default=METRIC_KEYS)
    p.set_defaults(func=aggregate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
