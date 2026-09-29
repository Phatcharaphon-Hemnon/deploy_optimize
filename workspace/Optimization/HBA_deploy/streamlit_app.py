"""Streamlit entrypoint: HBA comparison dashboard.

Run locally with::

    streamlit run streamlit_app.py

The dashboard never imports ``main.py`` (which executes an experiment on
import). Objectives come from ``_tool.py`` via ``hba_experiment`` exactly as
implemented there; see the notes on local implementations in the app.
"""

import matplotlib.pyplot as plt
import streamlit as st

import charts
import hba_experiment
from hba_experiment import DEFAULTS, FORMULAS, FUNCTION_ORDER, KNOWN_OPTIMUM, LIMITS, MINIMIZER_NOTES

st.set_page_config(page_title="HBA comparison", layout="wide")
st.title("Honey Badger Algorithm — comparison across local objectives")
st.caption(
    "Configurable HBA runs using the update equations from `main.py`. "
    "Comparisons are results for these local implementations (see Function details)."
)

LOCAL_IMPL_NOTE = (
    "These are results for the local implementations in `_tool.py`, not necessarily "
    "the textbook benchmarks of the same names: `paraboloid` uses only the last "
    "coordinate, `powell_sum` computes weighted squares, and `schwefel` computes "
    "an absolute sum."
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


def _run_form():
    with st.form("experiment_form"):
        st.subheader("Experiment configuration")
        functions = st.multiselect(
            "Functions", FUNCTION_ORDER, default=list(DEFAULTS["functions"])
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
                "Master seed", min_value=0, max_value=2**32 - 1,
                value=DEFAULTS["seed"], step=1,
            )
        submitted = st.form_submit_button("Run experiment")
    return submitted, {
        "functions": functions,
        "agents": int(agents),
        "dimensions": int(dimensions),
        "iterations": int(iterations),
        "repetitions": int(repetitions),
        "lb": float(lb),
        "ub": float(ub),
        "seed": int(seed),
    }


def _execute(cfg):
    try:
        names = hba_experiment.validate_functions(cfg["functions"])
        hba_experiment.validate_counts(
            cfg["agents"], cfg["dimensions"], cfg["iterations"],
            cfg["repetitions"], names,
        )
        hba_experiment.validate_bounds(cfg["lb"], cfg["ub"], names)
        hba_experiment.validate_seed(cfg["seed"])
    except ValueError as exc:
        st.error(str(exc))
        return
    progress = st.progress(0.0)
    status = st.empty()
    status.info("Running optimization…")
    try:
        results, settings = hba_experiment.run_experiment(
            names, agents=cfg["agents"], dimensions=cfg["dimensions"],
            iterations=cfg["iterations"], repetitions=cfg["repetitions"],
            lb=cfg["lb"], ub=cfg["ub"], seed=cfg["seed"],
            progress_callback=progress.progress,
        )
    except ValueError as exc:
        status.empty()
        progress.empty()
        st.error(str(exc))
        return
    computed = charts.compute_all(results, KNOWN_OPTIMUM)
    st.session_state["hba_results"] = results
    st.session_state["hba_settings"] = settings
    st.session_state["hba_computed"] = computed
    progress.progress(1.0)
    status.success(
        f"Completed: {len(names)} function(s) × {settings.repetitions} repetition(s), "
        f"{settings.iterations} iterations. Seeds {settings.seeds[0]}–{settings.seeds[-1]} "
        "(same schedule across functions)."
    )


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
        fig = charts.plot_combined_mean_error(computed)
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
    name = st.selectbox("Function", names, key="detail_function")
    entry = computed[name]
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
    fig_avg = charts.plot_average_best(entry["per_iteration"], f"{name}: average-best fitness")
    st.pyplot(fig_avg)
    plt.close(fig_avg)
    st.subheader("Optimum error")
    error_scale = st.radio(
        "Error axis", ["symlog", "linear"], index=0, horizontal=True,
        key="detail_scale",
        help="Symmetric-log preserves exact zeros with a linear region below 1e-12.",
    )
    fig_err = charts.plot_error(entry["per_iteration"], f"{name}: error", scale=error_scale)
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


def _downloads_tab(results, settings, computed):
    st.caption("All files are generated in memory from the retained results; downloading never reruns optimization.")
    st.download_button(
        "settings/seeds JSON",
        charts.settings_json(settings),
        file_name="settings.json",
        mime="application/json",
        key="dl_settings",
    )
    for name in FUNCTION_ORDER:
        if name not in results:
            continue
        entry = computed[name]
        runs = results[name]
        st.markdown(f"**{name}**")
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
            fig_avg = charts.plot_average_best(entry["per_iteration"], f"{name}: average-best fitness")
            st.download_button(
                "average-best PNG", charts.figure_png(fig_avg),
                file_name=f"{name}_average_best.png", mime="image/png",
                key=f"dl_avg_{name}",
            )
            plt.close(fig_avg)
            fig_err = charts.plot_error(entry["per_iteration"], f"{name}: error", scale="symlog")
            st.download_button(
                "error PNG (symlog)", charts.figure_png(fig_err),
                file_name=f"{name}_error_symlog.png", mime="image/png",
                key=f"dl_errlog_{name}",
            )
            plt.close(fig_err)
            fig_lin = charts.plot_error(entry["per_iteration"], f"{name}: error", scale="linear")
            st.download_button(
                "error PNG (linear)", charts.figure_png(fig_lin),
                file_name=f"{name}_error_linear.png", mime="image/png",
                key=f"dl_errlin_{name}",
            )
            plt.close(fig_lin)
        with col3:
            st.empty()
    st.markdown("**Combined archive**")
    figures = {}
    fig_comb = charts.plot_combined_mean_error(computed)
    figures["combined_mean_error.png"] = charts.figure_png(fig_comb)
    plt.close(fig_comb)
    for name in results:
        entry = computed[name]
        fig_avg = charts.plot_average_best(entry["per_iteration"], f"{name}: average-best fitness")
        figures[f"{name}_average_best.png"] = charts.figure_png(fig_avg)
        plt.close(fig_avg)
        fig_err = charts.plot_error(entry["per_iteration"], f"{name}: error", scale="symlog")
        figures[f"{name}_error_symlog.png"] = charts.figure_png(fig_err)
        plt.close(fig_err)
    st.download_button(
        "Download all (ZIP)", charts.build_zip(results, computed, settings, figures),
        file_name="hba_results.zip", mime="application/zip", key="dl_zip",
    )


def main():
    _init_state()
    submitted, cfg = _run_form()
    if submitted:
        _execute(cfg)
    results = st.session_state["hba_results"]
    settings = st.session_state["hba_settings"]
    computed = st.session_state["hba_computed"]
    if results is None:
        st.info(
            "Configure the experiment above and press **Run experiment**. "
            f"Defaults: all six functions, {DEFAULTS['agents']} agents, "
            f"{DEFAULTS['dimensions']} dimensions, {DEFAULTS['iterations']} iterations, "
            f"{DEFAULTS['repetitions']} repetitions, bounds "
            f"[{DEFAULTS['lb']:g}, {DEFAULTS['ub']:g}], seed {DEFAULTS['seed']}."
        )
        return
    tab_cmp, tab_fn, tab_dl = st.tabs(["Comparison", "Function details", "Downloads"])
    with tab_cmp:
        _comparison_tab(results, settings, computed)
    with tab_fn:
        _function_tab(results, settings, computed)
    with tab_dl:
        _downloads_tab(results, settings, computed)


main()
