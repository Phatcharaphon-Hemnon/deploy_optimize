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


def build_zip(results, computed, settings, figures, extra=None, extra_files=None):
    """Build a ZIP archive in memory with all CSV/JSON/PNG artifacts.

    ``extra`` is merged into ``settings.json`` (e.g. dashboard provenance)
    and ``extra_files`` maps additional archive names to bytes/str content
    (e.g. landscape slice metadata).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("settings.json", settings_json(settings, extra))
        for name, runs in results.items():
            entry = computed[name]
            zf.writestr(f"{name}/statistics.csv", statistics_csv(name, entry["per_iteration"]))
            zf.writestr(f"{name}/histories.csv", histories_csv(name, runs, 0.0))
            zf.writestr(f"{name}/runs.csv", runs_csv(name, runs, 0.0))
        for arcname, data in figures.items():
            zf.writestr(arcname, data)
        for arcname, data in (extra_files or {}).items():
            zf.writestr(arcname, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Objective landscapes (dimension-aware slices for the dashboard)
# ---------------------------------------------------------------------------

#: Fixed grid resolution for landscape evaluation.
LANDSCAPE_GRID = 100


def evaluate_landscape_1d(func, lb, ub, n=LANDSCAPE_GRID):
    """Evaluate ``func`` on ``n`` points spanning ``[lb, ub]``.

    Returns ``(xs, ys)``; non-finite evaluations become NaN.
    """
    n = int(n)
    xs = np.linspace(float(lb), float(ub), n)
    ys = np.empty(n, dtype=float)
    for k, x in enumerate(xs):
        try:
            value = float(func(np.array([x], dtype=float)))
        except Exception:
            value = float("nan")
        ys[k] = value if np.isfinite(value) else float("nan")
    return xs, ys


def evaluate_landscape_2d(func, dim, lb, ub, ix, iy, fixed, n=LANDSCAPE_GRID):
    """Evaluate ``func`` on an ``n x n`` grid over coordinates ``ix``, ``iy``.

    Remaining coordinates are held at ``fixed`` (a length-``dim`` vector).
    Returns ``(X, Y, Z)``; non-finite evaluations become NaN.
    """
    dim, ix, iy, n = int(dim), int(ix), int(iy), int(n)
    if not (0 <= ix < dim and 0 <= iy < dim):
        raise ValueError(f"Coordinate indices out of range for {dim} dimensions.")
    if ix == iy:
        raise ValueError("Landscape needs two distinct coordinates.")
    fixed = np.asarray(fixed, dtype=float)
    if fixed.shape != (dim,):
        raise ValueError("Fixed-position vector must match dimensions.")
    xs = np.linspace(float(lb), float(ub), n)
    ys = np.linspace(float(lb), float(ub), n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X, dtype=float)
    for j in range(n):
        for i in range(n):
            vec = fixed.copy()
            vec[ix] = X[j, i]
            vec[iy] = Y[j, i]
            try:
                value = float(func(vec))
            except Exception:
                value = float("nan")
            Z[j, i] = value if np.isfinite(value) else float("nan")
    return X, Y, Z


def _finite_values(Z, context):
    finite = np.asarray(Z, dtype=float)[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError(f"No finite objective values on this landscape grid ({context}).")
    return finite


def _is_constant(finite):
    return bool(np.ptp(finite) == 0)


def plot_landscape_line(xs, ys, title, xlabel="x0"):
    """Objective-versus-coordinate line plot for one-dimensional slices."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ys)
    if not np.any(mask):
        raise ValueError("No finite objective values on this landscape grid (line).")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs[mask], ys[mask])
    if _is_constant(ys[mask]):
        ax.text(0.5, 0.5, f"Constant landscape: f = {float(ys[mask][0]):g}",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Objective fitness")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_landscape_contour(X, Y, Z, title, xlabel="x0", ylabel="x1"):
    """Filled-contour view of a 2D objective slice."""
    finite = _finite_values(Z, "contour")
    fig, ax = plt.subplots(figsize=(7, 5))
    masked = np.ma.masked_invalid(Z)
    extent = (float(X.min()), float(X.max()), float(Y.min()), float(Y.max()))
    if _is_constant(finite):
        ax.imshow(masked, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.text(0.5, 0.5, f"Constant landscape: f = {float(finite[0]):g}",
                transform=ax.transAxes, ha="center", va="center")
    else:
        contour = ax.contourf(X, Y, masked, levels=50, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Objective fitness")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_landscape_surface(X, Y, Z, title, xlabel="x0", ylabel="x1"):
    """3D surface view of a 2D objective slice (height = fitness)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

    finite = _finite_values(Z, "surface")
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, np.ma.masked_invalid(Z), cmap="viridis",
                    edgecolor="none", alpha=0.9)
    if _is_constant(finite):
        ax.text2D(0.5, 0.5, f"Constant landscape: f = {float(finite[0]):g}",
                  transform=ax.transAxes, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Objective fitness (height)")
    ax.set_title(title + "  [height = fitness]")
    fig.tight_layout()
    return fig


def landscape_meta(function_name, dim, varied, fixed_values, rep, lb, ub,
                   grid_n=LANDSCAPE_GRID):
    """Slice metadata for display labels and ZIP archives."""
    return {
        "function": function_name,
        "dimensions": int(dim),
        "varied_coordinates": [f"x{i}" for i in varied] if varied else ["x0"],
        "fixed_coordinates": {f"x{k}": float(v) for k, v in fixed_values.items()},
        "slice_repetition": rep,
        "bounds": [float(lb), float(ub)],
        "grid": f"{int(grid_n)}x{int(grid_n)}" if varied and len(varied) == 2 else str(int(grid_n)),
    }


# ---------------------------------------------------------------------------
# Moving-agent animations (Plotly frames; convergence charts stay static)
# ---------------------------------------------------------------------------


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for agent animations. Install it with "
            "`pip install plotly` (see requirements.txt)."
        ) from exc
    import plotly.graph_objects as go

    return go


def _anim_title(function_name, repetition, seed, iteration, best_fitness):
    return (
        f"{function_name} — rep {repetition}, seed {seed}, "
        f"iteration {int(iteration)}, best fitness {float(best_fitness):.6g}"
    )


def _playback_controls(frame_names, frame_ms):
    """Shared Play/Pause buttons and iteration slider for Plotly animations."""
    frame_ms = int(frame_ms)
    buttons = [
        {
            "label": "Play",
            "method": "animate",
            "args": [
                None,
                {
                    "frame": {"duration": frame_ms, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                    "mode": "immediate",
                },
            ],
        },
        {
            "label": "Pause",
            "method": "animate",
            "args": [
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        },
    ]
    steps = [
        {
            "label": str(nm),
            "method": "animate",
            "args": [
                [nm],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        }
        for nm in frame_names
    ]
    sliders = [{"steps": steps, "currentvalue": {"prefix": "Iteration: "}}]
    return buttons, sliders


def build_animation_1d(xs, ys, trajectory, function_name, repetition, seed,
                       lb, ub, frame_ms=300):
    """Animate agents on the 1D objective-versus-coordinate curve."""
    go = _require_plotly()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pops = np.asarray(trajectory.populations, dtype=float)
    bests = np.asarray(trajectory.best_positions, dtype=float)
    best_fit = np.asarray(trajectory.best_fitness, dtype=float)
    frame_iters = [int(v) for v in np.asarray(trajectory.frame_iterations).tolist()]
    if pops.ndim != 3 or pops.shape[2] != 1:
        raise ValueError("1D animation needs populations with shape (frames, agents, 1).")
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        raise ValueError("No finite objective values on this landscape grid (line).")
    ylo, yhi = float(finite.min()), float(finite.max())
    if ylo == yhi:
        ylo, yhi = ylo - 1.0, yhi + 1.0
    pad = 0.05 * (yhi - ylo)
    ylo, yhi = ylo - pad, yhi + pad
    mask = np.isfinite(ys)
    it0 = frame_iters[0]
    agents_x = pops[0, :, 0]
    agents_y = np.asarray(trajectory.population_fitness, dtype=float)[0]
    best_x = float(bests[0, 0])
    best_y = float(best_fit[0])
    title0 = _anim_title(function_name, repetition, seed, it0, best_y)
    fig = go.Figure(
        data=[
            go.Scatter(x=xs[mask].tolist(), y=ys[mask].tolist(), mode="lines",
                       name="Objective f(x0)"),
            go.Scatter(x=agents_x.tolist(), y=agents_y.tolist(), mode="markers",
                       name="Agents", marker={"size": 7}),
            go.Scatter(x=[best_x], y=[best_y], mode="markers",
                       name="Best", marker={"size": 12, "symbol": "star"}),
        ],
        layout={
            "title": title0,
            "xaxis": {"range": [float(lb), float(ub)], "title": "x0"},
            "yaxis": {"range": [ylo, yhi], "title": "Objective fitness"},
        },
        frames=[
            go.Frame(
                name=str(it),
                data=[
                    go.Scatter(x=pops[k, :, 0].tolist(),
                               y=np.asarray(trajectory.population_fitness, dtype=float)[k].tolist()),
                    go.Scatter(x=[float(bests[k, 0])], y=[float(best_fit[k])]),
                ],
                layout={"title": _anim_title(function_name, repetition, seed, it, float(best_fit[k]))},
            )
            for k, it in enumerate(frame_iters)
        ],
    )
    buttons, sliders = _playback_controls([str(v) for v in frame_iters], frame_ms)
    fig.update_layout(updatemenus=[{"type": "buttons", "buttons": buttons}], sliders=sliders)
    return fig


def build_animation_contour(X, Y, Z, trajectory, ix, iy, function_name,
                            repetition, seed, lb, ub, frame_ms=300,
                            projected=True):
    """Animate agent projections on a fixed contour slice."""
    go = _require_plotly()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    pops = np.asarray(trajectory.populations, dtype=float)
    bests = np.asarray(trajectory.best_positions, dtype=float)
    best_fit = np.asarray(trajectory.best_fitness, dtype=float)
    pop_fit = np.asarray(trajectory.population_fitness, dtype=float)
    frame_iters = [int(v) for v in np.asarray(trajectory.frame_iterations).tolist()]
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError("No finite objective values on this landscape grid (contour).")
    zmin, zmax = float(finite.min()), float(finite.max())
    agent_label = (f"Agents (x{ix}×x{iy} projection)" if projected else "Agents")
    best_label = (f"Best (x{ix}×x{iy} projection)" if projected else "Best")
    it0 = frame_iters[0]
    ax, ay = pops[0, :, ix].tolist(), pops[0, :, iy].tolist()
    hover0 = [f"agent {i}<br>actual fitness {float(pop_fit[0, i]):.6g}" for i in range(len(ax))]
    fig = go.Figure(
        data=[
            go.Contour(x=X[0].tolist(), y=Y[:, 0].tolist(), z=Z.tolist(),
                       colorscale="Viridis", zmin=zmin, zmax=zmax,
                       name="Objective slice", showscale=True,
                       colorbar={"title": "Objective fitness (slice)"}),
            go.Scatter(x=ax, y=ay, mode="markers", name=agent_label,
                       marker={"size": 7}, text=hover0, hoverinfo="text"),
            go.Scatter(x=[float(bests[0, ix])], y=[float(bests[0, iy])],
                       mode="markers", name=best_label,
                       marker={"size": 12, "symbol": "star"},
                       text=[f"best<br>actual fitness {float(best_fit[0]):.6g}"],
                       hoverinfo="text"),
        ],
        layout={
            "title": _anim_title(function_name, repetition, seed, it0, float(best_fit[0])),
            "xaxis": {"range": [float(lb), float(ub)], "title": f"x{ix}"},
            "yaxis": {"range": [float(lb), float(ub)], "title": f"x{iy}"},
        },
        frames=[
            go.Frame(
                name=str(it),
                data=[
                    go.Scatter(
                        x=pops[k, :, ix].tolist(), y=pops[k, :, iy].tolist(),
                        text=[f"agent {i}<br>actual fitness {float(pop_fit[k, i]):.6g}"
                              for i in range(pops.shape[1])],
                    ),
                    go.Scatter(
                        x=[float(bests[k, ix])], y=[float(bests[k, iy])],
                        text=[f"best<br>actual fitness {float(best_fit[k]):.6g}"],
                    ),
                ],
                layout={"title": _anim_title(function_name, repetition, seed, it, float(best_fit[k]))},
            )
            for k, it in enumerate(frame_iters)
        ],
    )
    buttons, sliders = _playback_controls([str(v) for v in frame_iters], frame_ms)
    fig.update_layout(updatemenus=[{"type": "buttons", "buttons": buttons}], sliders=sliders)
    return fig


def build_animation_surface(X, Y, Z, trajectory, ix, iy, fixed, func,
                            function_name, repetition, seed, lb, ub,
                            frame_ms=300, projected=True):
    """Animate agent projections above a fixed 3D surface slice.

    Marker heights are the slice evaluation at the projected coordinates with
    the remaining coordinates fixed; hover text reports the actual
    full-dimensional fitness.
    """
    go = _require_plotly()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    pops = np.asarray(trajectory.populations, dtype=float)
    bests = np.asarray(trajectory.best_positions, dtype=float)
    best_fit = np.asarray(trajectory.best_fitness, dtype=float)
    pop_fit = np.asarray(trajectory.population_fitness, dtype=float)
    frame_iters = [int(v) for v in np.asarray(trajectory.frame_iterations).tolist()]
    fixed = np.asarray(fixed, dtype=float)
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError("No finite objective values on this landscape grid (surface).")

    def _slice_height(vec2x, vec2y):
        vec = fixed.copy()
        vec[ix] = float(vec2x)
        vec[iy] = float(vec2y)
        try:
            value = float(func(vec))
        except Exception:
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    zmin, zmax = float(finite.min()), float(finite.max())
    if zmin == zmax:
        zmin, zmax = zmin - 1.0, zmax + 1.0
    n_agents = pops.shape[1]
    ax0 = pops[0, :, ix].tolist()
    ay0 = pops[0, :, iy].tolist()
    az0 = [_slice_height(x, y) for x, y in zip(ax0, ay0)]
    bx0, by0 = float(bests[0, ix]), float(bests[0, iy])
    bz0 = _slice_height(bx0, by0)
    agent_label = (f"Agents (x{ix}×x{iy} projection)" if projected else "Agents")
    best_label = (f"Best (x{ix}×x{iy} projection)" if projected else "Best")
    it0 = frame_iters[0]
    fig = go.Figure(
        data=[
            go.Surface(x=X.tolist(), y=Y.tolist(), z=Z.tolist(),
                       colorscale="Viridis", cmin=zmin, cmax=zmax,
                       name="Objective slice",
                       colorbar={"title": "Objective fitness (slice)"}),
            go.Scatter3d(x=ax0, y=ay0, z=az0, mode="markers", name=agent_label,
                         marker={"size": 4},
                         text=[f"agent {i}<br>slice height {float(az0[i]):.6g}<br>"
                               f"actual fitness {float(pop_fit[0, i]):.6g}" for i in range(n_agents)],
                         hoverinfo="text"),
            go.Scatter3d(x=[bx0], y=[by0], z=[bz0], mode="markers", name=best_label,
                         marker={"size": 7, "symbol": "diamond"},
                         text=[f"best<br>slice height {float(bz0):.6g}<br>"
                               f"actual fitness {float(best_fit[0]):.6g}"],
                         hoverinfo="text"),
        ],
        layout={
            "title": _anim_title(function_name, repetition, seed, it0, float(best_fit[0])),
            "scene": {
                "xaxis": {"range": [float(lb), float(ub)], "title": f"x{ix}"},
                "yaxis": {"range": [float(lb), float(ub)], "title": f"x{iy}"},
                "zaxis": {"range": [zmin, zmax], "title": "Objective fitness (slice height)"},
            },
        },
        frames=[
            go.Frame(
                name=str(it),
                data=[
                    go.Scatter3d(
                        x=pops[k, :, ix].tolist(), y=pops[k, :, iy].tolist(),
                        z=[_slice_height(x, y) for x, y in
                           zip(pops[k, :, ix].tolist(), pops[k, :, iy].tolist())],
                        text=[f"agent {i}<br>actual fitness {float(pop_fit[k, i]):.6g}"
                              for i in range(n_agents)],
                    ),
                    go.Scatter3d(
                        x=[float(bests[k, ix])], y=[float(bests[k, iy])],
                        z=[_slice_height(float(bests[k, ix]), float(bests[k, iy]))],
                        text=[f"best<br>actual fitness {float(best_fit[k]):.6g}"],
                    ),
                ],
                layout={"title": _anim_title(function_name, repetition, seed, it, float(best_fit[k]))},
            )
            for k, it in enumerate(frame_iters)
        ],
    )
    buttons, sliders = _playback_controls([str(v) for v in frame_iters], frame_ms)
    fig.update_layout(updatemenus=[{"type": "buttons", "buttons": buttons}], sliders=sliders)
    return fig


def animation_html(fig):
    """Standalone interactive HTML for one animation figure."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def animation_meta(function_name, dim, kind, varied, fixed_values, rep, seed,
                   frames, lb, ub, grid_n=LANDSCAPE_GRID):
    """Metadata describing one generated agent animation."""
    return {
        "function": function_name,
        "dimensions": int(dim),
        "kind": kind,
        "varied_coordinates": [f"x{i}" for i in varied],
        "fixed_coordinates": {f"x{k}": float(v) for k, v in fixed_values.items()},
        "record_repetition": int(rep),
        "record_seed": int(seed),
        "frames": int(frames),
        "bounds": [float(lb), float(ub)],
        "grid": f"{int(grid_n)}x{int(grid_n)}" if len(varied) == 2 else str(int(grid_n)),
        "note": ("Markers are coordinate projections; surface heights are slice "
                 "evaluations while hover text reports actual full-dimensional fitness."),
    }
