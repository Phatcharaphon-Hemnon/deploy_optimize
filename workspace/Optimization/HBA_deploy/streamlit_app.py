"""Streamlit entrypoint: HBA comparison dashboard.

Run locally with::

    streamlit run streamlit_app.py

The dashboard never imports ``main.py`` (which executes an experiment on
import). Objectives come from ``_tool.py`` via ``hba_experiment`` exactly as
implemented there; see the notes on local implementations in the app.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import charts
import hba_experiment

#: Names ``streamlit_app.py`` requires from the engine module. Validated
#: before use so a stale/duplicated deployment fails with the loaded path and
#: the missing names instead of a bare ImportError. No fallbacks, no reloads.
REQUIRED_ENGINE_EXPORTS = (
    "DEFAULTS", "DISPLAY_NAMES", "FORMULAS", "FUNCTION_ORDER",
    "KNOWN_OPTIMUM", "LIMITS", "MINIMIZER_NOTES", "MODEL_REVISION",
)

_missing_engine_exports = [
    name for name in REQUIRED_ENGINE_EXPORTS if not hasattr(hba_experiment, name)
]
if _missing_engine_exports:
    import logging as _logging

    _logging.error(
        "Deployment version mismatch: loaded hba_experiment from %s; "
        "missing exports: %s",
        getattr(hba_experiment, "__file__", "?"), _missing_engine_exports,
    )
    raise ImportError(
        "Deployment version mismatch: the loaded hba_experiment module "
        f"({getattr(hba_experiment, '__file__', '?')}) is missing "
        f"{sorted(_missing_engine_exports)}. Redeploy with matching app, "
        "experiment, charts, and objective sources (commit 19446ae or later) "
        "and reboot the app (a page refresh alone keeps imported modules)."
    )
del _missing_engine_exports

from hba_experiment import DEFAULTS, DISPLAY_NAMES, FORMULAS, FUNCTION_ORDER, KNOWN_OPTIMUM, LIMITS, MINIMIZER_NOTES, MODEL_REVISION

FUNCTION_OPTIONS = ["All functions"] + FUNCTION_ORDER

st.set_page_config(page_title="HBA comparison", layout="wide")
st.title("Honey Badger Algorithm — comparison across local objectives")
st.caption(
    "Configurable HBA runs using the standard Honey Badger update equations "
    "(digging/honey phases with density factor `alpha = 2*exp(-t/T)`). "
    "Comparisons are results for these local implementations (see Function details)."
)

def _display(name):
    return DISPLAY_NAMES.get(name, name)


LOCAL_IMPL_NOTE = (
    "These are results for the local implementations in `_tool.py`: `paraboloid` "
    "sums squares over every coordinate (Paraboloid/Sphere), `powell_sum` is the "
    "Sum of Different Powers, and `schwefel` computes an absolute sum (Absolute "
    "Sum/L1), not the commonly named Schwefel 2.26 objective."
)
SCALE_NOTE = (
    "Function scales differ, so lower raw errors on one function than on another "
    "do not establish an overall ranking."
)
OPTIMUM_NOTE = (
    f"All six local objectives have known optimum value {KNOWN_OPTIMUM:g} when the "
    "bounds contain a known minimizer (coordinate 1 for Rosenbrock, coordinate 0 "
    "for the others)."
)


def _init_state():
    for key in ("hba_results", "hba_settings", "hba_computed"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "hba_run_id" not in st.session_state:
        st.session_state["hba_run_id"] = 0
    if "hba_seed_auto" not in st.session_state:
        st.session_state["hba_seed_auto"] = False
    for key in ("hba_landscapes", "hba_landscape_pngs", "hba_landscape_meta",
                "hba_animation_htmls", "hba_animation_meta"):
        if key not in st.session_state:
            st.session_state[key] = {}
    if "hba_zip_bytes" not in st.session_state:
        st.session_state["hba_zip_bytes"] = None
    if "hba_zip_error" not in st.session_state:
        st.session_state["hba_zip_error"] = None
    if "hba_zip_prepared_for" not in st.session_state:
        st.session_state["hba_zip_prepared_for"] = None
    if "hba_model_revision" not in st.session_state:
        st.session_state["hba_model_revision"] = MODEL_REVISION


def _settings_revision_current(settings):
    """True when retained settings match the current objective/optimizer model."""
    return getattr(settings, "model_revision", None) == MODEL_REVISION


def _clear_results_state():
    st.session_state["hba_results"] = None
    st.session_state["hba_settings"] = None
    st.session_state["hba_computed"] = None
    st.session_state["hba_landscapes"] = {}
    st.session_state["hba_landscape_pngs"] = {}
    st.session_state["hba_landscape_meta"] = {}
    st.session_state["hba_animation_htmls"] = {}
    st.session_state["hba_animation_meta"] = {}
    st.session_state["hba_zip_bytes"] = None
    st.session_state["hba_zip_error"] = None
    st.session_state["hba_zip_prepared_for"] = None


def _run_form():
    with st.form("experiment_form"):
        st.subheader("Experiment configuration")
        functions = st.multiselect(
            "Functions", FUNCTION_ORDER, default=list(FUNCTION_ORDER),
            format_func=_display,
            help="Select any subset of the six local objectives; at least one is required.",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            agents = st.number_input(
                "Agents", min_value=1, max_value=LIMITS["agents"],
                value=DEFAULTS["agents"], step=1,
            )
            dimensions = st.number_input(
                "Dimensions", min_value=1, max_value=LIMITS["dimensions"],
                value=DEFAULTS["dimensions"], step=1,
            )
        with col2:
            iterations = st.number_input(
                "Iterations", min_value=1, max_value=LIMITS["iterations"],
                value=DEFAULTS["iterations"], step=1,
            )
            repetitions = st.number_input(
                "Repetitions", min_value=1, max_value=LIMITS["repetitions"],
                value=DEFAULTS["repetitions"], step=1,
            )
        with col3:
            lb = st.number_input("Lower bound", value=float(DEFAULTS["lb"]))
            ub = st.number_input("Upper bound", value=float(DEFAULTS["ub"]))
            seed = st.number_input(
                "Master seed (optional)", min_value=0, max_value=2**32 - 1,
                value=None, step=1,
                help="Leave blank for a random seed drawn at submission; "
                     "an explicit seed (including 0) reproduces results.",
            )
        record_animation = st.checkbox(
            "Record agent animation", value=True,
            help="When enabled, one repetition per selected function records "
                 "agent trajectories for the moving-agent animation. "
                 "Recording copies populations only and never alters results.",
        )
        record_rep = st.number_input(
            "Repetition to record (1-indexed)", min_value=1,
            max_value=LIMITS["repetitions"], value=1, step=1,
            help="Which repetition to record for each selected function (default 1).",
        )
        submitted = st.form_submit_button("Run experiment")
    return submitted, {
        "functions": list(functions),
        "agents": int(agents),
        "dimensions": int(dimensions),
        "iterations": int(iterations),
        "repetitions": int(repetitions),
        "lb": float(lb),
        "ub": float(ub),
        "seed": None if seed is None else int(seed),
        "record_animation": bool(record_animation),
        "record_rep": int(record_rep),
    }


def _execute(cfg):
    try:
        names = hba_experiment.validate_functions(cfg["functions"])
        hba_experiment.validate_counts(
            cfg["agents"], cfg["dimensions"], cfg["iterations"],
            cfg["repetitions"], names,
        )
        hba_experiment.validate_bounds(cfg["lb"], cfg["ub"], names)
        record_animation = bool(cfg.get("record_animation", False))
        record_rep = int(cfg.get("record_rep", 1))
        if record_animation:
            hba_experiment.validate_record_rep(record_rep, cfg["repetitions"])
            # Explain before execution when even two frames cannot fit in 64 MiB.
            hba_experiment.plan_trajectory_frames(
                cfg["iterations"], cfg["agents"], cfg["dimensions"], len(names))
        if cfg["seed"] is None:
            import secrets

            resolved_seed = secrets.randbelow(2**32)
            seed_auto = True
        else:
            resolved_seed = hba_experiment.validate_seed(cfg["seed"])
            seed_auto = False
    except ValueError as exc:
        st.error(str(exc))
        return False
    progress = st.progress(0.0)
    status = st.empty()
    status.info("Running optimization…")
    try:
        results, settings = hba_experiment.run_experiment(
            names, agents=cfg["agents"], dimensions=cfg["dimensions"],
            iterations=cfg["iterations"], repetitions=cfg["repetitions"],
            lb=cfg["lb"], ub=cfg["ub"], seed=resolved_seed,
            progress_callback=progress.progress,
            record_trajectories=record_animation, record_rep=record_rep,
        )
    except ValueError as exc:
        status.empty()
        progress.empty()
        st.error(str(exc))
        return False
    computed = charts.compute_all(results, KNOWN_OPTIMUM)
    st.session_state["hba_run_id"] += 1
    st.session_state["hba_results"] = results
    st.session_state["hba_settings"] = settings
    st.session_state["hba_computed"] = computed
    st.session_state["hba_seed_auto"] = seed_auto
    # Stale artifacts belong to the previous experiment.
    st.session_state["hba_landscapes"] = {}
    st.session_state["hba_landscape_pngs"] = {}
    st.session_state["hba_landscape_meta"] = {}
    st.session_state["hba_animation_htmls"] = {}
    st.session_state["hba_animation_meta"] = {}
    st.session_state["hba_zip_bytes"] = None
    st.session_state["hba_zip_error"] = None
    st.session_state["hba_zip_prepared_for"] = None
    st.session_state["hba_model_revision"] = MODEL_REVISION
    progress.progress(1.0)
    origin = "drawn at submission" if seed_auto else "explicit"
    status.success(
        f"Completed: {len(names)} function(s) × {settings.repetitions} repetition(s), "
        f"{settings.iterations} iterations. Master seed {settings.seed} ({origin}); "
        f"repetition seeds {settings.seeds[0]}–{settings.seeds[-1]} "
        "(same schedule across functions)."
    )
    return True


def _summary_rows(computed):
    rows = []
    for name in FUNCTION_ORDER:
        if name not in computed:
            continue
        final = computed[name]["final"]
        rows.append({
            "function": name,
            "mean fitness": final["mean_fitness"],
            "median fitness": final["median_fitness"],
            "std fitness": "n/a" if final["std_fitness"] is None else final["std_fitness"],
            "min fitness": final["min_fitness"],
            "max fitness": final["max_fitness"],
            "mean error": final["mean_error"],
            "median error": final["median_error"],
            "std error": "n/a" if final["std_error"] is None else final["std_error"],
            "min error": final["min_error"],
            "max error": final["max_error"],
        })
    return rows


def _comparison_tab(results, settings, computed):
    st.info(LOCAL_IMPL_NOTE)
    st.info(SCALE_NOTE + " " + OPTIMUM_NOTE)
    error_scale = st.radio(
        "Error axis", ["symlog", "linear"], index=0, horizontal=True,
        key="comparison_scale",
        help="Symmetric-log preserves exact zeros with a linear region below 1e-12.",
    )
    if error_scale == "symlog":
        fig = charts.plot_combined_mean_error(computed, display_names=DISPLAY_NAMES)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, entry in computed.items():
            xs = range(len(entry["per_iteration"]["mean_error"]))
            ax.plot(xs, entry["per_iteration"]["mean_error"], label=name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean abs error (linear)")
        ax.set_title("Mean error across functions")
        ax.legend()
        fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Final best-solution statistics (sortable)")
    st.caption(
        "Mean/median/min/max over repetitions of final best fitness and error. "
        "Standard deviation is the sample standard deviation (ddof=1); "
        "it is unavailable ('n/a') with a single repetition."
    )
    st.dataframe(_summary_rows(computed), use_container_width=True)


def _function_tab(results, settings, computed):
    names = list(results)
    name = st.selectbox("Function", names, key="detail_function", format_func=_display)
    entry = computed[name]
    st.markdown(f"**{_display(name)}**")
    st.markdown(f"**Formula (local implementation):** `{FORMULAS[name]}`")
    st.markdown(f"**Minimizer:** {MINIMIZER_NOTES[name]}")
    if settings.repetitions < 2:
        st.caption("Single repetition: variability bands are omitted; standard deviation is unavailable.")
    st.subheader("Average-best convergence")
    st.caption(
        "Mean and median of the runs' best-so-far fitness at each iteration "
        "(iteration 0 is the best initialized agent), with a mean ± one "
        "sample-standard-deviation band."
    )
    fig_avg = charts.plot_average_best(entry["per_iteration"], f"{_display(name)}: average-best fitness")
    st.pyplot(fig_avg)
    plt.close(fig_avg)
    st.subheader("Optimum error")
    error_scale = st.radio(
        "Error axis", ["symlog", "linear"], index=0, horizontal=True,
        key="detail_scale",
        help="Symmetric-log preserves exact zeros with a linear region below 1e-12.",
    )
    fig_err = charts.plot_error(entry["per_iteration"], f"{_display(name)}: error", scale=error_scale)
    st.pyplot(fig_err)
    plt.close(fig_err)
    st.subheader("Best positions per repetition")
    runs = results[name]
    pos_rows = []
    for r, run in enumerate(runs):
        row = {"run": r, "seed": run.seed, "final fitness": float(run.best_fitness),
               "final error": float(abs(float(run.best_fitness) - KNOWN_OPTIMUM))}
        for i, v in enumerate(run.best_position):
            row[f"x{i}"] = float(v)
        pos_rows.append(row)
    st.dataframe(pos_rows, use_container_width=True)
    _landscape_section(name, results, settings)
    _animation_section(name, results, settings)


def _cached_grid(grid_key, compute):
    store = st.session_state["hba_landscapes"]
    if grid_key not in store:
        store[grid_key] = compute()
    return store[grid_key]


def _landscape_png(png_key, fig):
    store = st.session_state["hba_landscape_pngs"]
    if png_key not in store:
        store[png_key] = charts.figure_png(fig)
    return store[png_key]


def _show_landscape_fig(make, png_key, download_label, file_name, dl_key, meta_key, meta):
    try:
        fig = make()
    except ValueError as exc:
        st.warning(str(exc))
        return
    try:
        png = _landscape_png(png_key, fig)
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)
    st.download_button(
        download_label, png, file_name=file_name, mime="image/png", key=dl_key,
    )
    st.session_state["hba_landscape_meta"][meta_key] = meta


def _render_landscape_kind(kind, grid, labels, name, png_prefix, meta):
    run_id = st.session_state["hba_run_id"]
    if kind == "line":
        xs, ys = grid
        title = f"{_display(name)}: objective vs {labels['varied']}"
        _show_landscape_fig(
            lambda: charts.plot_landscape_line(xs, ys, title, xlabel=labels["varied"]),
            (run_id, png_prefix, "line"),
            "Landscape PNG (line)", f"{png_prefix}_landscape_line.png",
            f"dl_land_{png_prefix}_line", (run_id, png_prefix, "line"), meta,
        )
        return
    X, Y, Z = grid
    if kind == "contour":
        title = f"{_display(name)}: contour over {labels['varied']}"
        make = lambda: charts.plot_landscape_contour(
            X, Y, Z, title, xlabel=labels["x"], ylabel=labels["y"])
    else:
        title = f"{_display(name)}: surface over {labels['varied']}"
        make = lambda: charts.plot_landscape_surface(
            X, Y, Z, title, xlabel=labels["x"], ylabel=labels["y"])
    _show_landscape_fig(
        make, (run_id, png_prefix, kind),
        f"Landscape PNG ({kind})", f"{png_prefix}_landscape_{kind}.png",
        f"dl_land_{png_prefix}_{kind}", (run_id, png_prefix, kind), meta,
    )


def _landscape_section(name, results, settings):
    run_id = st.session_state["hba_run_id"]
    dim = settings.dimensions
    st.subheader("Objective landscape")
    st.caption("Surface height represents objective fitness, not a third optimization coordinate.")
    show = st.checkbox(
        "Show objective landscape", value=False, key=f"land_show_r{run_id}_{name}",
    )
    if not show:
        return
    func = hba_experiment.FUNCTIONS[name]
    lb, ub = settings.lb, settings.ub
    if dim == 1:
        grid_key = (run_id, name, "line")
        grid = _cached_grid(
            grid_key, lambda: charts.evaluate_landscape_1d(func, lb, ub))
        meta = charts.landscape_meta(name, dim, [0], {}, None, lb, ub,
                               display_name=_display(name))
        st.markdown("**Varied:** `x0`. Full 1D objective landscape.")
        _render_landscape_kind("line", grid, {"varied": "x0"}, name,
                               f"{name}_d1", meta)
        return
    view = st.radio(
        "Landscape view", ["Both", "2D contour", "3D surface"], index=0,
        horizontal=True, key=f"land_view_r{run_id}_{name}",
    )
    kinds = ["contour", "surface"] if view == "Both" else (
        ["contour"] if view == "2D contour" else ["surface"])
    if dim == 2:
        grid_key = (run_id, name, "full")
        grid = _cached_grid(
            grid_key,
            lambda: charts.evaluate_landscape_2d(
                func, dim, lb, ub, 0, 1, np.zeros(dim)),
        )
        meta = charts.landscape_meta(name, dim, [0, 1], {}, None, lb, ub,
                                 display_name=_display(name))
        st.markdown("**Varied:** `x0`, `x1`. Full 2D objective landscape.")
        labels = {"varied": "`x0` × `x1`", "x": "x0", "y": "x1"}
        prefix = f"{name}_x0_x1"
        for kind in kinds:
            _render_landscape_kind(kind, grid, labels, name, prefix, meta)
        return
    coord_options = [f"x{i}" for i in range(dim)]
    cx_label = st.selectbox(
        "X coordinate", coord_options, index=0, key=f"land_cx_r{run_id}_{name}")
    cy_label = st.selectbox(
        "Y coordinate", coord_options, index=1, key=f"land_cy_r{run_id}_{name}")
    ix, iy = int(cx_label[1:]), int(cy_label[1:])
    if ix == iy:
        st.error("Select two distinct coordinates for the landscape slice.")
        return
    rep = st.selectbox(
        "Slice repetition (best position holds fixed coordinates)",
        list(range(1, settings.repetitions + 1)), index=0,
        key=f"land_rep_r{run_id}_{name}",
    )
    fixed = np.asarray(results[name][rep - 1].best_position, dtype=float)
    grid_key = (run_id, name, ix, iy, rep)
    grid = _cached_grid(
        grid_key,
        lambda: charts.evaluate_landscape_2d(func, dim, lb, ub, ix, iy, fixed),
    )
    fixed_values = {k: float(fixed[k]) for k in range(dim) if k not in (ix, iy)}
    meta = charts.landscape_meta(name, dim, [ix, iy], fixed_values, rep, lb, ub,
                             display_name=_display(name))
    fixed_text = ", ".join(f"`x{k}={v:g}`" for k, v in fixed_values.items())
    st.markdown(
        f"**Varied:** `{cx_label}` × `{cy_label}`. "
        f"**Fixed** at repetition {rep}'s best position: {fixed_text}.")
    labels = {"varied": f"`{cx_label}` × `{cy_label}`", "x": cx_label, "y": cy_label}
    prefix = f"{name}_{cx_label}_{cy_label}_rep{rep}"
    for kind in kinds:
        _render_landscape_kind(kind, grid, labels, name, prefix, meta)


SPEED_TO_MS = {"0.5×": 600, "1×": 300, "2×": 150, "4×": 75}


def _recorded_trajectory(name, results, settings):
    rep = int(getattr(settings, "record_rep", 1))
    if not bool(getattr(settings, "record_animation", False)):
        return None, rep
    runs = results.get(name, [])
    if rep < 1 or rep > len(runs):
        return None, rep
    return runs[rep - 1].trajectory, rep


def _animation_section(name, results, settings):
    run_id = st.session_state["hba_run_id"]
    dim = settings.dimensions
    st.subheader("Animation")
    st.caption("Convergence charts above remain static; only this section animates agents.")
    trajectory, rep = _recorded_trajectory(name, results, settings)
    if trajectory is None:
        st.info(
            "Agent animation was not recorded for this run."
        )
        st.caption(
            "Recovery reruns the complete experiment (all functions × repetitions) "
            "with the saved settings and resolved master seed, recording repetition 1."
        )
        if st.button("Rerun with animation", key=f"rerun_anim_r{run_id}_{name}"):
            cfg = {
                "functions": list(settings.functions),
                "agents": int(settings.agents),
                "dimensions": int(settings.dimensions),
                "iterations": int(settings.iterations),
                "repetitions": int(settings.repetitions),
                "lb": float(settings.lb),
                "ub": float(settings.ub),
                "seed": int(settings.seed),
                "record_animation": True,
                "record_rep": 1,
            }
            if _execute(cfg):
                st.rerun()
        return
    n_frames = int(np.asarray(trajectory.frame_iterations).shape[0])
    st.caption(
        f"Recorded repetition {rep} (seed {int(trajectory.seed)}): "
        f"{n_frames} frames from iteration {int(trajectory.frame_iterations[0])} "
        f"to {int(trajectory.frame_iterations[-1])}, including initialization and final iteration."
    )
    func = hba_experiment.FUNCTIONS[name]
    lb, ub = settings.lb, settings.ub
    if dim == 1:
        speed = st.selectbox(
            "Playback speed", list(SPEED_TO_MS), index=1,
            key=f"anim_speed_r{run_id}_{name}",
            help="Frame duration only; playback never reruns optimization.",
        )
        frame_ms = SPEED_TO_MS[speed]
        grid = _cached_grid(
            (run_id, name, "line"),
            lambda: charts.evaluate_landscape_1d(func, lb, ub))
        xs, ys = grid
        try:
            fig = charts.build_animation_1d(
                xs, ys, trajectory, name, rep, int(trajectory.seed),
                lb, ub, frame_ms=frame_ms, display_name=_display(name))
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))
            return
        st.plotly_chart(fig, use_container_width=True, key=f"anim_plot_r{run_id}_{name}_line")
        html = charts.animation_html(fig)
        meta = charts.animation_meta(name, dim, "line", [0], {}, rep,
                                     int(trajectory.seed), n_frames, lb, ub,
                                     display_name=_display(name))
        st.session_state["hba_animation_htmls"][(run_id, name, "line")] = html
        st.session_state["hba_animation_meta"][(run_id, name, "line")] = meta
        st.download_button(
            "Animation HTML (standalone)", html,
            file_name=f"{name}_animation_line.html", mime="text/html",
            key=f"dl_anim_{name}_line",
        )
        return
    ctl_speed, ctl_view = st.columns(2)
    with ctl_speed:
        speed = st.selectbox(
            "Playback speed", list(SPEED_TO_MS), index=1,
            key=f"anim_speed_r{run_id}_{name}",
            help="Frame duration only; playback never reruns optimization.",
        )
    with ctl_view:
        kind = st.radio(
            "Animation view", ["2D contour", "3D surface"], index=0, horizontal=True,
            key=f"anim_view_r{run_id}_{name}",
            help="Only the selected animation is generated on demand; grids stay static during playback.",
        )
    frame_ms = SPEED_TO_MS[speed]
    kind_key = "contour" if kind == "2D contour" else "surface"
    if dim == 2:
        ix, iy = 0, 1
        fixed = np.zeros(dim)
        fixed_values = {}
        grid = _cached_grid(
            (run_id, name, "full"),
            lambda: charts.evaluate_landscape_2d(func, dim, lb, ub, 0, 1, np.zeros(dim)),
        )
        st.markdown("**Varied:** `x0` × `x1`. Full 2D objective landscape; markers show true positions.")
    else:
        coord_options = [f"x{i}" for i in range(dim)]
        sel_x, sel_y = st.columns(2)
        with sel_x:
            cx_label = st.selectbox(
                "Animation X coordinate", coord_options, index=0,
                key=f"anim_cx_r{run_id}_{name}")
        with sel_y:
            cy_label = st.selectbox(
                "Animation Y coordinate", coord_options, index=1,
                key=f"anim_cy_r{run_id}_{name}")
        ix, iy = int(cx_label[1:]), int(cy_label[1:])
        if ix == iy:
            st.error("Select two distinct coordinates for the animation slice.")
            return
        # Fixed slice through the recorded repetition's final best position.
        fixed = np.asarray(results[name][rep - 1].best_position, dtype=float)
        fixed_values = {k: float(fixed[k]) for k in range(dim) if k not in (ix, iy)}
        grid = _cached_grid(
            (run_id, name, ix, iy, rep),
            lambda: charts.evaluate_landscape_2d(func, dim, lb, ub, ix, iy, fixed),
        )
        fixed_text = ", ".join(f"`x{k}={v:g}`" for k, v in fixed_values.items())
        st.markdown(
            f"**Varied:** `{cx_label}` × `{cy_label}` (coordinate projections). "
            f"**Fixed** at recorded repetition {rep}'s final best position: {fixed_text}. "
            "Surface marker heights are slice evaluations; hover text reports actual full-dimensional fitness.")
    X, Y, Z = grid
    try:
        if kind_key == "contour":
            fig = charts.build_animation_contour(
                X, Y, Z, trajectory, ix, iy, name, rep, int(trajectory.seed),
                lb, ub, frame_ms=frame_ms, projected=(dim > 2),
                display_name=_display(name))
        else:
            fig = charts.build_animation_surface(
                X, Y, Z, trajectory, ix, iy, fixed, func, name, rep,
                int(trajectory.seed), lb, ub, frame_ms=frame_ms,
                projected=(dim > 2), display_name=_display(name))
    except (ValueError, ImportError) as exc:
        st.warning(str(exc))
        return
    st.plotly_chart(fig, use_container_width=True, key=f"anim_plot_r{run_id}_{name}_{kind_key}_{ix}_{iy}")
    html = charts.animation_html(fig)
    meta = charts.animation_meta(name, dim, kind_key, [ix, iy], fixed_values, rep,
                                 int(trajectory.seed), n_frames, lb, ub,
                                 display_name=_display(name))
    akey = (run_id, name, kind_key) if dim == 2 else (run_id, name, kind_key, ix, iy)
    st.session_state["hba_animation_htmls"][akey] = html
    st.session_state["hba_animation_meta"][akey] = meta
    st.download_button(
        f"Animation HTML ({kind}, standalone)", html,
        file_name=f"{name}_animation_{kind_key}.html", mime="text/html",
        key=f"dl_anim_{name}_{kind_key}_{ix}_{iy}",
    )


def _zip_fingerprint(run_id, land_pngs, anim_htmls):
    land_keys = sorted(
        f"{p}/{k}" for (rid, p, *rest) in [kk for kk in land_pngs]
        for k in [rest[-1]] if rid == run_id
    ) if land_pngs else []
    anim_keys = sorted("/".join(str(v) for v in kk[1:]) for kk in anim_htmls if kk and kk[0] == run_id)
    return (run_id, tuple(land_keys), tuple(anim_keys))


def _downloads_tab(results, settings, computed):
    st.caption("All files are generated in memory from the retained results; downloading never reruns optimization.")
    seed_extra = {"master_seed_auto": bool(st.session_state["hba_seed_auto"])}
    st.download_button(
        "settings/seeds JSON",
        charts.settings_json(settings, seed_extra),
        file_name="settings.json",
        mime="application/json",
        key="dl_settings",
    )
    for name in FUNCTION_ORDER:
        if name not in results:
            continue
        entry = computed[name]
        runs = results[name]
        st.markdown(f"**{_display(name)}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "statistics CSV", charts.statistics_csv(name, entry["per_iteration"]),
                file_name=f"{name}_statistics.csv", mime="text/csv",
                key=f"dl_stats_{name}",
            )
            st.download_button(
                "histories CSV", charts.histories_csv(name, runs, KNOWN_OPTIMUM),
                file_name=f"{name}_histories.csv", mime="text/csv",
                key=f"dl_hist_{name}",
            )
            st.download_button(
                "best positions CSV", charts.runs_csv(name, runs, KNOWN_OPTIMUM),
                file_name=f"{name}_runs.csv", mime="text/csv",
                key=f"dl_runs_{name}",
            )
        with col2:
            fig_avg = charts.plot_average_best(entry["per_iteration"], f"{_display(name)}: average-best fitness")
            st.download_button(
                "average-best PNG", charts.figure_png(fig_avg),
                file_name=f"{name}_average_best.png", mime="image/png",
                key=f"dl_avg_{name}",
            )
            plt.close(fig_avg)
            fig_err = charts.plot_error(entry["per_iteration"], f"{_display(name)}: error", scale="symlog")
            st.download_button(
                "error PNG (symlog)", charts.figure_png(fig_err),
                file_name=f"{name}_error_symlog.png", mime="image/png",
                key=f"dl_errlog_{name}",
            )
            plt.close(fig_err)
            fig_lin = charts.plot_error(entry["per_iteration"], f"{_display(name)}: error", scale="linear")
            st.download_button(
                "error PNG (linear)", charts.figure_png(fig_lin),
                file_name=f"{name}_error_linear.png", mime="image/png",
                key=f"dl_errlin_{name}",
            )
            plt.close(fig_lin)
        with col3:
            st.empty()
    st.markdown("**Combined archive**")
    st.caption(
        "Press **Prepare ZIP** to build the archive from the retained results; "
        "downloading never reruns optimization. Preparing again is required "
        "after the experiment or its landscapes/animations change."
    )
    run_id = st.session_state["hba_run_id"]
    land_pngs = st.session_state["hba_landscape_pngs"]
    land_meta = st.session_state["hba_landscape_meta"]
    anim_htmls = st.session_state["hba_animation_htmls"]
    anim_meta = st.session_state["hba_animation_meta"]
    if land_pngs:
        st.markdown("**Generated landscape images**")
    for (rid, prefix, kind), png in sorted(
            land_pngs.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][2]))):
        if rid != run_id:
            continue
        st.download_button(
            f"Landscape PNG ({prefix}, {kind})", png,
            file_name=f"{prefix}_landscape_{kind}.png", mime="image/png",
            key=f"dl_landzip_{prefix}_{kind}",
        )
    run_anim_keys = [kk for kk in anim_htmls if kk and kk[0] == run_id]
    if run_anim_keys:
        st.markdown("**Generated animations**")
        for kk in sorted(run_anim_keys, key=str):
            html = anim_htmls[kk]
            label = "/".join(str(v) for v in kk[1:])
            st.download_button(
                f"Animation HTML ({label}, standalone)", html,
                file_name=f"{'_'.join(str(v) for v in kk[1:])}_animation.html",
                mime="text/html", key=f"dl_animzip_{'_'.join(str(v) for v in kk[1:])}",
            )
    fingerprint = _zip_fingerprint(run_id, land_pngs, anim_htmls)
    # Clear prepared archives whenever the experiment or included artifacts change.
    if st.session_state["hba_zip_prepared_for"] != fingerprint:
        st.session_state["hba_zip_bytes"] = None
        st.session_state["hba_zip_error"] = None
    if st.button("Prepare ZIP", key="prepare_zip"):
        import inspect as _inspect
        import json as _json
        import traceback as _tb

        try:
            figures = {}
            fig_comb = charts.plot_combined_mean_error(computed)
            figures["combined_mean_error.png"] = charts.figure_png(fig_comb)
            plt.close(fig_comb)
            for name in results:
                entry = computed[name]
                fig_avg = charts.plot_average_best(entry["per_iteration"], f"{_display(name)}: average-best fitness")
                figures[f"{name}_average_best.png"] = charts.figure_png(fig_avg)
                plt.close(fig_avg)
                fig_err = charts.plot_error(entry["per_iteration"], f"{_display(name)}: error", scale="symlog")
                figures[f"{name}_error_symlog.png"] = charts.figure_png(fig_err)
                plt.close(fig_err)
            for (rid, prefix, kind), png in sorted(
                    land_pngs.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][2]))):
                if rid != run_id:
                    continue
                figures[f"landscapes/{prefix}_landscape_{kind}.png"] = png
            land_entries = []
            for (rid, prefix, kind) in sorted(
                    [kk for kk in land_pngs if kk[0] == run_id], key=str):
                land_entries.append({
                    "file": f"landscapes/{prefix}_landscape_{kind}.png",
                    "png_key": f"{prefix}/{kind}",
                    "slice": land_meta.get((rid, prefix, kind), {}),
                })
            extra_files = {}
            if land_entries:
                extra_files["landscapes/slices.json"] = _json.dumps(land_entries, indent=2)
            anim_entries = []
            for kk in sorted(run_anim_keys, key=str):
                arcname = f"animations/{'_'.join(str(v) for v in kk[1:])}_animation.html"
                extra_files[arcname] = anim_htmls[kk]
                anim_entries.append({"file": arcname, "anim_key": "/".join(str(v) for v in kk[1:]),
                                     "animation": anim_meta.get(kk, {})})
            if anim_entries:
                extra_files["animations/metadata.json"] = _json.dumps(anim_entries, indent=2)
            # Diagnose deployment drift: report the imported module path/signature
            # alongside any failure instead of silently discarding optionals.
            sig = str(_inspect.signature(charts.build_zip))
            _ = (charts.__file__, sig)
            data = charts.build_zip(results, computed, settings, figures,
                                    extra=seed_extra, extra_files=extra_files)
            st.session_state["hba_zip_bytes"] = data
            st.session_state["hba_zip_prepared_for"] = fingerprint
            st.session_state["hba_zip_error"] = None
        except Exception as exc:  # contained within Downloads
            st.session_state["hba_zip_bytes"] = None
            st.session_state["hba_zip_error"] = str(exc)
            import logging as _logging

            _logging.exception("ZIP preparation failed")
            st.error(
                "Could not build the ZIP archive: "
                f"{exc}. Completed results, plots, and individual downloads below "
                "remain available. "
                f"(charts={getattr(charts, '__file__', '?')}, "
                f"build_zip{str(_inspect.signature(charts.build_zip))})"
            )
            with st.expander("Traceback (for deployment diagnosis)"):
                st.code(_tb.format_exc())
    if st.session_state["hba_zip_error"] and st.session_state["hba_zip_bytes"] is None:
        st.warning(f"Last ZIP preparation failed: {st.session_state['hba_zip_error']}")
    if st.session_state["hba_zip_bytes"] is not None:
        st.download_button(
            "Download prepared ZIP",
            st.session_state["hba_zip_bytes"],
            file_name="hba_results.zip", mime="application/zip", key="dl_zip",
        )
    else:
        st.info("No prepared archive yet — press **Prepare ZIP** above.")


def main():
    _init_state()
    submitted, cfg = _run_form()
    if submitted:
        _execute(cfg)
    results = st.session_state["hba_results"]
    settings = st.session_state["hba_settings"]
    computed = st.session_state["hba_computed"]
    if results is not None and not _settings_revision_current(settings):
        # Objective/optimizer model changed since these results were produced:
        # clear everything derived from them and request a new run (no auto-rerun).
        _clear_results_state()
        results = None
        st.warning(
            "Previously completed results were produced by an older model revision "
            f"(expected revision {MODEL_REVISION}) and have been cleared. "
            "Press **Run experiment** for a new run with the current formulas."
        )
    if results is None:
        st.info(
            "Configure the experiment above and press **Run experiment**. "
            f"Defaults: all six functions, {DEFAULTS['agents']} agents, "
            f"{DEFAULTS['dimensions']} dimensions, {DEFAULTS['iterations']} iterations, "
            f"{DEFAULTS['repetitions']} repetitions, bounds "
            f"[{DEFAULTS['lb']:g}, {DEFAULTS['ub']:g}], blank master seed (random)."
        )
        return
    origin = "drawn at submission" if st.session_state["hba_seed_auto"] else "explicit"
    st.caption(
        f"Completed experiment: {len(results)} function(s) × "
        f"{settings.repetitions} repetition(s), {settings.iterations} iterations, "
        f"{settings.dimensions}D, bounds [{settings.lb:g}, {settings.ub:g}]. "
        f"Master seed {settings.seed} ({origin}); repetition seeds "
        f"{settings.seeds[0]}–{settings.seeds[-1]} (same schedule across functions). "
        "Changing chart, landscape, or download options never reruns optimization."
    )
    tab_cmp, tab_fn, tab_dl = st.tabs(["Comparison", "Function details", "Downloads"])
    with tab_cmp:
        _comparison_tab(results, settings, computed)
    with tab_fn:
        _function_tab(results, settings, computed)
    with tab_dl:
        _downloads_tab(results, settings, computed)


main()
