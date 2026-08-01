"""
benchmark_runtime_scalability.py

Empirically measures how surgical_embeddings.generate_embeddings() scales with
input size across models, and plots the result in the same log-log style as
the "Runtime scalability concept" panel.

Usage:
    python benchmark_runtime_scalability.py

Requires:
    pip install surgical-embeddings matplotlib numpy
    (or: pip install -e ".[test]" from a local clone of the repo)
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.surgical_embeddings import (
    generate_embeddings,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = ["MiniLM", "SapBERT", "BGE_Large"]
SIZES = [10, 100, 1000, 10000]
REPEATS = 5          # number of timed trials per (model, size) pair
APPLY_PCA = False     # keep PCA off so timings isolate the embedding step
WARMUP_TERM = ["warmup term for cache priming"]

# A small pool of real surgical terms to sample from when building each
# input size. Expand this list with real vocabulary (e.g. CPT/ICD procedure
# names) for more representative timing -- token length affects runtime.
TERM_POOL = [
    "laparoscopic appendectomy",
    "total knee arthroplasty",
    "robotic prostatectomy",
    "coronary artery bypass graft",
    "open reduction internal fixation",
    "endoscopic sinus surgery",
    "carotid endarterectomy",
    "cholecystectomy",
    "hernia repair",
    "spinal fusion",
    "hip replacement",
    "mastectomy",
    "thoracotomy",
    "craniotomy",
    "nephrectomy",
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_terms(n: int) -> list[str]:
    """Build an input list of length n by cycling through the term pool."""
    return [TERM_POOL[i % len(TERM_POOL)] for i in range(n)]


def warm_up(model: str) -> None:
    """Trigger model download/cache load so it isn't counted in timing."""
    generate_embeddings(WARMUP_TERM, model_name=model, apply_pca=APPLY_PCA)


def time_one_run(terms: list[str], model: str) -> float:
    """Time a single call to generate_embeddings(); returns elapsed seconds."""
    start = time.perf_counter()
    generate_embeddings(terms, model_name=model, apply_pca=APPLY_PCA)
    return time.perf_counter() - start


def run_benchmark(
    models: list[str] = MODELS,
    sizes: list[int] = SIZES,
    repeats: int = REPEATS,
) -> dict:
    """
    Runs the timing sweep across models and input sizes.

    Returns a dict of the form:
        {
            "MiniLM": {10: {"median": ..., "trials": [...]}, 100: {...}, ...},
            "SapBERT": {...},
            "BGE_Large": {...},
        }
    """
    results: dict = {}

    for model in models:
        print(f"\n=== Warming up {model} ===")
        warm_up(model)

        results[model] = {}
        for n in sizes:
            terms = make_terms(n)
            trial_times = []
            for trial in range(repeats):
                elapsed = time_one_run(terms, model)
                trial_times.append(elapsed)
                print(f"{model} | n={n:>6} | trial {trial + 1}/{repeats} | {elapsed:.4f}s")

            results[model][n] = {
                "median": statistics.median(trial_times),
                "mean": statistics.mean(trial_times),
                "stdev": statistics.stdev(trial_times) if len(trial_times) > 1 else 0.0,
                "trials": trial_times,
            }

    return results


def fit_power_law(sizes: list[int], times: list[float]) -> tuple[float, float]:
    """
    Fits runtime ~= a * n^k via log-log linear regression.
    Returns (k, a).
    """
    log_n = np.log(np.array(sizes, dtype=float))
    log_t = np.log(np.array(times, dtype=float))
    k, log_a = np.polyfit(log_n, log_t, 1)
    return k, float(np.exp(log_a))


def save_results(results: dict, path: str = "runtime_scalability_results.json") -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_runtime_scalability(
    results: dict,
    output_path: str = "runtime_scalability.png",
    relative: bool = True,
    show_fit: bool = True,
) -> None:
    """
    Plots measured runtime vs. input size on log-log axes, in the same style
    as the "Runtime scalability concept" panel.

    Parameters
    ----------
    results : dict
        Output of run_benchmark().
    output_path : str
        Where to save the PNG.
    relative : bool
        If True, normalize each model's curve to its own value at the
        smallest input size (matches the "Relative runtime" y-axis in the
        original figure). If False, plot raw seconds.
    show_fit : bool
        If True, annotate each line with its fitted scaling exponent k
        from runtime ~= a * n^k.
    """
    colors = {
        "MiniLM": "#4C8577",
        "SapBERT": "#8E5B8C",
        "BGE_Large": "#5B5BA8",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for model, per_size in results.items():
        sizes_sorted = sorted(per_size.keys())
        medians = [per_size[n]["median"] for n in sizes_sorted]

        if relative:
            baseline = medians[0]
            y_values = [m / baseline for m in medians]
            y_label = "Relative runtime"
        else:
            y_values = medians
            y_label = "Runtime (seconds)"

        color = colors.get(model, None)
        ax.plot(
            sizes_sorted,
            y_values,
            marker="o",
            label=model,
            color=color,
            linewidth=2,
        )

        if show_fit:
            k, _ = fit_power_law(sizes_sorted, medians)
            print(f"{model}: fitted scaling exponent k = {k:.2f} (runtime ~ n^{k:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of input terms")
    ax.set_ylabel(y_label)
    ax.set_title("Runtime scalability (measured)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)
    plot_runtime_scalability(results, relative=True, show_fit=True)
    plot_runtime_scalability(
        results,
        output_path="runtime_scalability_absolute.png",
        relative=False,
        show_fit=True,
    )