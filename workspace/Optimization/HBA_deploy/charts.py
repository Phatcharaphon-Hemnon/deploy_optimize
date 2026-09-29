"""Statistics, charts, and in-memory exports for HBA comparison results.

Average-best is the mean of the runs' best-so-far fitness at each iteration
(not the mean agent fitness). Error is ``abs(best_fitness - known_optimum)``.
Shaded bands are mean +/- one sample standard deviation (``ddof=1``); with a
single repetition the standard deviation is unavailable and bands are
omitted.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LINTHRESH = 1e-12


def _as_histories(runs):
    return np.asarray([np.asarray(r.best_history, dtype=float) for r in runs], dtype=float)


def per_iteration_stats(histories, optimum=0.0):
    """Compute per-iteration fitness/error statistics.

    Returns a dict with ``mean_fitness``, ``median_fitness``, ``std_fitness``
    (NaN when fewer than two runs), ``min_fitness``, ``max_fitness`` and the
    same ``*_error`` series for ``abs(history - optimum)``.
    """
    histories = np.asarray(histories, dtype=float)
    if histories.ndim != 2:
        raise ValueError("histories must have shape (repetitions, iterations + 1).")
    if histories.shape[0] < 1:
        raise ValueError("Need at least one repetition.")
    if not np.all(np.isfinite(histories)):
        raise ValueError("histories must be finite.")
    errors = np.abs(histories - float(optimum))
    n = histories.shape[0]
    std_fit = np.std(histories, axis=0, ddof=1) if n >= 2 else np.full(histories.shape[1], np.nan)
    std_err = np.std(errors, axis=0, ddof=1) if n >= 2 else np.full(errors.shape[1], np.nan)
    return {
        "n": n,
        "mean_fitness": np.mean(histories, axis=0),
        "median_fitness": np.median(histories, axis=0),
        "std_fitness": std_fit,
        "min_fitness": np.min(histories, axis=0),
        "max_fitness": np.max(histories, axis=0),
        "mean_error": np.mean(errors, axis=0),
        "median_error": np.median(errors, axis=0),
        "std_error": std_err,
        "min_error": np.min(errors, axis=0),
        "max_error": np.max(errors, axis=0),
    }


def final_summary(histories, optimum=0.0):
    """Final-iteration mean, median, std (None if n == 1), min, max."""
    stats = per_iteration_stats(histories, optimum)
    n = stats["n"]
    out = {}
    for key in ("mean_fitness", "median_fitness", "min_fitness", "max_fitness",
                "mean_error", "median_error", "min_error", "max_error"):
        out[key] = float(stats[key][-1])
    out["std_fitness"] = None if n < 2 else float(stats["std_fitness"][-1])
    out["std_error"] = None if n < 2 else float(stats["std_error"][-1])
    out["n"] = n
    return out


def compute_all(results, optimum=0.0):
    """Map function name to ``{'histories', 'per_iteration', 'final'}``."""
    out = {}
    for name, runs in results.items():
        histories = _as_histories(runs)
        out[name] = {
            "histories": histories,
            "per_iteration": per_iteration_stats(histories, optimum),
            "final": final_summary(histories, optimum),
        }
    return out


def _band(ax, xs, mean, std, **kwargs):
    upper = mean + std
    lower = np.clip(mean - std, 0.0, None)  # display-only clipping
    ax.fill_between(xs, lower, upper, **kwargs)


def plot_average_best(per_iteration, title, xlabel="Iteration"):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(per_iteration["mean_fitness"]))
    n = per_iteration["n"]
    ax.plot(xs, per_iteration["mean_fitness"], label="Mean best-so-far")
    ax.plot(xs, per_iteration["median_fitness"], label="Median best-so-far", linestyle="--")
    if n >= 2:
        _band(ax, xs, per_iteration["mean_fitness"], per_iteration["std_fitness"],
              alpha=0.25, label="Mean ± sample std")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best-so-far fitness")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error(per_iteration, title, scale="symlog", xlabel="Iteration"):
    if scale not in ("linear", "symlog"):
        raise ValueError("scale must be 'linear' or 'symlog'.")
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(per_iteration["mean_error"]))
    n = per_iteration["n"]
    ax.plot(xs, per_iteration["mean_error"], label="Mean error")
    ax.plot(xs, per_iteration["median_error"], label="Median error", linestyle="--")
    if n >= 2:
        _band(ax, xs, per_iteration["mean_error"], per_iteration["std_error"],
              alpha=0.25, label="Mean ± sample std")
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("abs(best fitness - optimum)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_combined_mean_error(computed, title="Mean error across functions"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, entry in computed.items():
        xs = np.arange(len(entry["per_iteration"]["mean_error"]))
        ax.plot(xs, entry["per_iteration"]["mean_error"], label=name)
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean abs error (symlog)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# In-memory exports (isolated per session: no disk writes)
# ---------------------------------------------------------------------------


def _csv_string(header, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def statistics_csv(name, per_iteration):
    n = len(per_iteration["mean_fitness"])
    rows = []
    for t in range(n):
        rows.append([
            t,
            per_iteration["mean_fitness"][t],
            per_iteration["median_fitness"][t],
            "" if per_iteration["n"] < 2 else per_iteration["std_fitness"][t],
            per_iteration["min_fitness"][t],
            per_iteration["max_fitness"][t],
            per_iteration["mean_error"][t],
            per_iteration["median_error"][t],
            "" if per_iteration["n"] < 2 else per_iteration["std_error"][t],
            per_iteration["min_error"][t],
            per_iteration["max_error"][t],
        ])
    return _csv_string(
        ["iteration", "mean_fitness", "median_fitness", "std_fitness", "min_fitness",
         "max_fitness", "mean_error", "median_error", "std_error", "min_error", "max_error"],
        rows,
    )


def histories_csv(name, runs, optimum=0.0):
    rows = []
    for r, run in enumerate(runs):
        for t, value in enumerate(np.asarray(run.best_history, dtype=float)):
            rows.append([r, t, float(value), float(abs(float(value) - optimum))])
    return _csv_string(["run", "iteration", "best_fitness", "error"], rows)


def runs_csv(name, runs, optimum=0.0):
    dim = len(np.asarray(runs[0].best_position, dtype=float)) if runs else 0
    header = ["run", "seed", "final_fitness", "final_error"] + [f"x{i}" for i in range(dim)]
    rows = []
    for r, run in enumerate(runs):
        pos = np.asarray(run.best_position, dtype=float)
        rows.append([r, run.seed, float(run.best_fitness),
                     float(abs(float(run.best_fitness) - optimum))] + [float(v) for v in pos])
    return _csv_string(header, rows)


def settings_json(settings, extra=None):
    payload = settings.to_dict()
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2)


def figure_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def build_zip(results, computed, settings, figures):
    """Build a ZIP archive in memory with all CSV/JSON/PNG artifacts."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("settings.json", settings_json(settings))
        for name, runs in results.items():
            entry = computed[name]
            zf.writestr(f"{name}/statistics.csv", statistics_csv(name, entry["per_iteration"]))
            zf.writestr(f"{name}/histories.csv", histories_csv(name, runs, 0.0))
            zf.writestr(f"{name}/runs.csv", runs_csv(name, runs, 0.0))
        for arcname, data in figures.items():
            zf.writestr(arcname, data)
    return buf.getvalue()
