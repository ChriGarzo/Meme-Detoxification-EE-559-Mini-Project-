"""
Benchmark single-step inference time and CO2 for each compared system:

  llava_teacher   LLaVA-Next 7B  — explain + rewrite one meme (2 forward passes)
  detoxllm        DetoxLLM 7B    — rewrite one text (text-only baseline)
  bart_finetuned  BART-large 400M — rewrite one text (our model, 'full' condition)

Models are loaded and released sequentially so peak VRAM stays within 40 GB.
Each model runs N_WARMUP warmup passes (not timed), then N_BENCH timed passes
under a CodeCarbon tracker.  Mean, median, and std of wall-clock time are
reported together with estimated CO2 per inference.

Usage:
    python analysis/benchmark_single_inference.py \\
        --validation_jsonl /scratch/hmr_stage2_dataset/val.jsonl \\
        --checkpoint_dir   /scratch/hmr_stage2_phase2_full_checkpoint \\
        --hf_cache         /scratch/hf_cache \\
        --output_dir       /scratch/hmr_inference_benchmark

Optional flags:
    --skip_llava          skip LLaVA benchmark (saves ~20 min)
    --skip_detoxllm       skip DetoxLLM benchmark
    --n_warmup N          warmup passes per model (default: 3)
    --n_bench  N          timed passes per model  (default: 10)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from codecarbon import EmissionsTracker

# ------------------------------------------------------------------
# Path setup so imports work whether called from repo root or analysis/
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, debug: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "benchmark.log"),
            logging.StreamHandler(),
        ],
    )


def load_example(validation_jsonl: Path) -> Dict[str, Any]:
    """Load first record from the validation JSONL."""
    with open(validation_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                logger.info(
                    "Benchmark example — id=%s  text='%s...'",
                    rec.get("id"),
                    (rec.get("original_text") or "")[:60],
                )
                return rec
    raise ValueError(f"No records found in {validation_jsonl}")


# ------------------------------------------------------------------
# LLaVA benchmark
# ------------------------------------------------------------------

def benchmark_llava(
    example: Dict[str, Any],
    hf_cache: Optional[str],
    n_warmup: int,
    n_bench: int,
    output_dir: Path,
) -> Dict[str, Any]:
    from models.explainer import MemeExplainer

    logger.info("=" * 60)
    logger.info("Benchmarking LLaVA-Next 7B (explain + rewrite per meme)")

    image_path = example.get("image_path") or ""
    text = example.get("original_text") or example.get("text") or ""

    t_load_start = time.perf_counter()
    explainer = MemeExplainer(cache_dir=hf_cache)
    explainer.load_model()
    load_time_s = time.perf_counter() - t_load_start
    logger.info("LLaVA loaded in %.1f s", load_time_s)

    num_params = sum(p.numel() for p in explainer.model.parameters())

    def one_inference():
        explanation = explainer.explain(image_path, text, max_retries=1)
        rewrite = explainer.generate_rewrite(image_path, text, explanation)
        return rewrite

    logger.info("Running %d warmup passes...", n_warmup)
    for _ in range(n_warmup):
        one_inference()

    logger.info("Running %d timed passes with CodeCarbon...", n_bench)
    tracker = EmissionsTracker(
        log_level="warning",
        output_dir=str(output_dir),
        output_file="emissions_llava_bench.csv",
    )
    tracker.start()
    times_ms = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        one_inference()
        times_ms.append((time.perf_counter() - t0) * 1000)
        logger.info("  pass %d/%d  %.0f ms", i + 1, n_bench, times_ms[-1])
    total_co2_kg = tracker.stop() or 0.0

    del explainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = _make_result(
        system="llava_teacher",
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        num_params=num_params,
        load_time_s=load_time_s,
        times_ms=times_ms,
        total_co2_kg=total_co2_kg,
        n_bench=n_bench,
        note="explain() + generate_rewrite() per meme (2 forward passes)",
    )
    logger.info("LLaVA:  mean=%.0f ms  CO2/inference=%.2f μg", result["mean_ms"], result["co2_per_inference_ug"])
    return result


# ------------------------------------------------------------------
# DetoxLLM benchmark
# ------------------------------------------------------------------

def benchmark_detoxllm(
    example: Dict[str, Any],
    hf_cache: Optional[str],
    n_warmup: int,
    n_bench: int,
    output_dir: Path,
) -> Dict[str, Any]:
    from baselines.run_detoxllm_baseline import DetoxLLMBaseline

    logger.info("=" * 60)
    logger.info("Benchmarking DetoxLLM 7B (text-only rewrite)")

    text = example.get("original_text") or example.get("text") or ""

    t_load_start = time.perf_counter()
    baseline = DetoxLLMBaseline(hf_cache=hf_cache)
    load_time_s = time.perf_counter() - t_load_start
    logger.info("DetoxLLM loaded in %.1f s", load_time_s)

    num_params = sum(p.numel() for p in baseline.model.parameters())

    def one_inference():
        return baseline.detoxify(text)

    logger.info("Running %d warmup passes...", n_warmup)
    for _ in range(n_warmup):
        one_inference()

    logger.info("Running %d timed passes with CodeCarbon...", n_bench)
    tracker = EmissionsTracker(
        log_level="warning",
        output_dir=str(output_dir),
        output_file="emissions_detoxllm_bench.csv",
    )
    tracker.start()
    times_ms = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        one_inference()
        times_ms.append((time.perf_counter() - t0) * 1000)
        logger.info("  pass %d/%d  %.0f ms", i + 1, n_bench, times_ms[-1])
    total_co2_kg = tracker.stop() or 0.0

    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = _make_result(
        system="detoxllm",
        model_id="UBC-NLP/DetoxLLM-7B",
        num_params=num_params,
        load_time_s=load_time_s,
        times_ms=times_ms,
        total_co2_kg=total_co2_kg,
        n_bench=n_bench,
        note="detoxify() — text-only, no image",
    )
    logger.info("DetoxLLM: mean=%.0f ms  CO2/inference=%.2f μg", result["mean_ms"], result["co2_per_inference_ug"])
    return result


# ------------------------------------------------------------------
# BART finetuned benchmark
# ------------------------------------------------------------------

def benchmark_bart_finetuned(
    example: Dict[str, Any],
    checkpoint_dir: Path,
    hf_cache: Optional[str],
    n_warmup: int,
    n_bench: int,
    output_dir: Path,
) -> Dict[str, Any]:
    from models.rewriter import MemeRewriter

    logger.info("=" * 60)
    logger.info("Benchmarking BART-finetuned 400M (text rewrite, 'full' condition)")

    text = example.get("original_text") or example.get("text") or ""
    target_group = example.get("target_group") or "unknown"
    visual_evidence = example.get("visual_evidence") or ""
    implicit_meaning = example.get("implicit_meaning") or ""

    t_load_start = time.perf_counter()
    rewriter = MemeRewriter(model_name=str(checkpoint_dir), cache_dir=hf_cache)
    rewriter.load_model()
    load_time_s = time.perf_counter() - t_load_start
    logger.info("BART-finetuned loaded in %.1f s", load_time_s)

    num_params = sum(p.numel() for p in rewriter.model.parameters())

    def one_inference():
        return rewriter.rewrite(
            text=text,
            target_group=target_group,
            visual_evidence=visual_evidence,
            implicit_meaning=implicit_meaning,
            mode="full",
        )

    logger.info("Running %d warmup passes...", n_warmup)
    for _ in range(n_warmup):
        one_inference()

    logger.info("Running %d timed passes with CodeCarbon...", n_bench)
    tracker = EmissionsTracker(
        log_level="warning",
        output_dir=str(output_dir),
        output_file="emissions_bart_bench.csv",
    )
    tracker.start()
    times_ms = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        one_inference()
        times_ms.append((time.perf_counter() - t0) * 1000)
        logger.info("  pass %d/%d  %.0f ms", i + 1, n_bench, times_ms[-1])
    total_co2_kg = tracker.stop() or 0.0

    del rewriter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = _make_result(
        system="bart_finetuned",
        model_id="facebook/bart-large (LoRA fine-tuned, condition=full)",
        num_params=num_params,
        load_time_s=load_time_s,
        times_ms=times_ms,
        total_co2_kg=total_co2_kg,
        n_bench=n_bench,
        note="rewrite() — full condition, explanation fields from val.jsonl",
    )
    logger.info("BART ft: mean=%.0f ms  CO2/inference=%.2f μg", result["mean_ms"], result["co2_per_inference_ug"])
    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_result(
    system: str,
    model_id: str,
    num_params: int,
    load_time_s: float,
    times_ms: List[float],
    total_co2_kg: float,
    n_bench: int,
    note: str = "",
) -> Dict[str, Any]:
    arr = np.array(times_ms)
    co2_per_inference_kg = total_co2_kg / n_bench if n_bench > 0 else 0.0
    return {
        "system": system,
        "model_id": model_id,
        "num_parameters": num_params,
        "num_parameters_M": round(num_params / 1e6, 1),
        "load_time_s": round(load_time_s, 2),
        "n_warmup": len(times_ms),  # stored for reference, actually n_bench used
        "n_bench": n_bench,
        "mean_ms": round(float(arr.mean()), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "std_ms": round(float(arr.std()), 1),
        "min_ms": round(float(arr.min()), 1),
        "max_ms": round(float(arr.max()), 1),
        "total_co2_kg_for_bench": total_co2_kg,
        "co2_per_inference_kg": co2_per_inference_kg,
        "co2_per_inference_ug": round(co2_per_inference_kg * 1e9, 4),  # nanograms → micrograms
        "co2_per_358_examples_mg": round(co2_per_inference_kg * 358 * 1e6, 4),
        "note": note,
    }


def print_benchmark_table(results: List[Dict[str, Any]]) -> None:
    col_w = [16, 12, 12, 12, 12, 14, 14]
    headers = ["System", "Params (M)", "Load (s)", "Mean (ms)", "Std (ms)", "CO2/inf (μg)", "CO2/358ex (mg)"]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  ".join("-" * w for w in col_w)

    print("\n=== Single-Inference Benchmark ===")
    print(row_fmt.format(*headers))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r["system"][:16],
            f"{r['num_parameters_M']:.0f}",
            f"{r['load_time_s']:.1f}",
            f"{r['mean_ms']:.0f}",
            f"{r['std_ms']:.0f}",
            f"{r['co2_per_inference_ug']:.4f}",
            f"{r['co2_per_358_examples_mg']:.4f}",
        ))
    print()

    # Speedup vs LLaVA
    llava = next((r for r in results if r["system"] == "llava_teacher"), None)
    if llava and len(results) > 1:
        print("Relative inference time vs LLaVA-Next:")
        for r in results:
            if r["system"] != "llava_teacher" and llava["mean_ms"] > 0:
                speedup = llava["mean_ms"] / r["mean_ms"]
                print(f"  {r['system']:20s}  {speedup:.0f}× faster")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark single-step inference time and CO2")
    parser.add_argument(
        "--validation_jsonl", type=Path,
        default=Path("/scratch/hmr_stage2_dataset/val.jsonl"),
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path,
        default=Path("/scratch/hmr_stage2_phase2_full_checkpoint"),
        help="Path to fine-tuned BART checkpoint (condition=full by default)",
    )
    parser.add_argument("--hf_cache", type=str, default="/scratch/hf_cache")
    parser.add_argument("--output_dir", type=Path, default=Path("/scratch/hmr_inference_benchmark"))
    parser.add_argument("--n_warmup", type=int, default=3, help="Warmup passes (not timed)")
    parser.add_argument("--n_bench", type=int, default=10, help="Timed passes")
    parser.add_argument("--skip_llava", action="store_true")
    parser.add_argument("--skip_detoxllm", action="store_true")
    parser.add_argument("--skip_bart", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    setup_logging(args.output_dir, debug=args.debug)
    logger.info("Single-inference benchmark starting")
    logger.info("  validation_jsonl : %s", args.validation_jsonl)
    logger.info("  checkpoint_dir   : %s", args.checkpoint_dir)
    logger.info("  hf_cache         : %s", args.hf_cache)
    logger.info("  n_warmup / n_bench: %d / %d", args.n_warmup, args.n_bench)

    example = load_example(args.validation_jsonl)

    results: List[Dict[str, Any]] = []

    if not args.skip_llava:
        try:
            results.append(benchmark_llava(example, args.hf_cache, args.n_warmup, args.n_bench, args.output_dir))
        except Exception as exc:
            logger.error("LLaVA benchmark failed: %s", exc, exc_info=True)

    if not args.skip_detoxllm:
        try:
            results.append(benchmark_detoxllm(example, args.hf_cache, args.n_warmup, args.n_bench, args.output_dir))
        except Exception as exc:
            logger.error("DetoxLLM benchmark failed: %s", exc, exc_info=True)

    if not args.skip_bart:
        try:
            results.append(benchmark_bart_finetuned(
                example, args.checkpoint_dir, args.hf_cache, args.n_warmup, args.n_bench, args.output_dir
            ))
        except Exception as exc:
            logger.error("BART-finetuned benchmark failed: %s", exc, exc_info=True)

    if not results:
        logger.error("No benchmark results produced.")
        return 1

    out_json = args.output_dir / "inference_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results written to %s", out_json)

    print_benchmark_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
