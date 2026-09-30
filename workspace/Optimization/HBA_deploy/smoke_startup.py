"""Startup smoke check for the Streamlit deployment.

Run from this directory in a fresh process::

    python3 smoke_startup.py

It imports the exact names ``streamlit_app.py`` requires from
``hba_experiment`` and ``charts``, asserts the corrected model metadata, and
runs a small recorded experiment through every chart/export builder the
dashboard uses. Missing imports or symbols fail explicitly here (no fallback
defaults), so a stale or mismatched module is caught at startup instead of at
first click.
"""

import io
import zipfile

import streamlit  # noqa: F401  (dashboard dependency must exist)

import charts
import hba_experiment
from hba_experiment import (
    DEFAULTS,
    DISPLAY_NAMES,
    FORMULAS,
    FUNCTION_ORDER,
    KNOWN_OPTIMUM,
    LIMITS,
    MINIMIZER_NOTES,
    MODEL_REVISION,
)

# Exact attribute surface used by streamlit_app.py; getattr raises ImportError-
# style AttributeError explicitly when a deployed module is stale/mismatched.
_CHART_NAMES = [
    "animation_html",
    "animation_meta",
    "build_animation_1d",
    "build_animation_contour",
    "build_animation_surface",
    "build_zip",
    "compute_all",
    "evaluate_landscape_1d",
    "evaluate_landscape_2d",
    "figure_png",
    "histories_csv",
    "landscape_meta",
    "plot_average_best",
    "plot_combined_mean_error",
    "plot_error",
    "plot_landscape_contour",
    "plot_landscape_line",
    "plot_landscape_surface",
    "runs_csv",
    "settings_json",
    "statistics_csv",
]
_ENGINE_NAMES = [
    "FUNCTIONS",
    "plan_trajectory_frames",
    "run_experiment",
    "validate_bounds",
    "validate_counts",
    "validate_functions",
    "validate_record_rep",
    "validate_seed",
]

for _name in _CHART_NAMES:
    getattr(charts, _name)
for _name in _ENGINE_NAMES:
    getattr(hba_experiment, _name)

assert MODEL_REVISION == 2, f"expected model revision 2, got {MODEL_REVISION!r}"
assert set(DISPLAY_NAMES) == set(FUNCTION_ORDER), "display labels must cover every function"
assert DISPLAY_NAMES["paraboloid"] == "Paraboloid (Sphere)"
assert DISPLAY_NAMES["powell_sum"] == "Sum of Different Powers"
assert DISPLAY_NAMES["schwefel"] == "Absolute Sum (L1)"
assert set(FORMULAS) == set(FUNCTION_ORDER)
assert set(MINIMIZER_NOTES) == set(FUNCTION_ORDER)
assert set(LIMITS) == set(DEFAULTS) - {"functions", "seed", "lb", "ub"}


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    from hba_experiment import KNOWN_OPTIMUM as _OPT

    results, settings = hba_experiment.run_experiment(
        ["paraboloid", "schwefel"], agents=4, dimensions=2, iterations=6,
        repetitions=2, lb=-5.0, ub=5.0, seed=123,
        record_trajectories=True, record_rep=1,
    )
    assert settings.to_dict()["model_revision"] == 2
    computed = charts.compute_all(results, _OPT)

    figures = {}
    fig = charts.plot_combined_mean_error(computed, display_names=DISPLAY_NAMES)
    legend_obj = fig.axes[0].get_legend()
    assert legend_obj is not None, "combined chart must have a legend"
    legend = sorted(t.get_text() for t in legend_obj.get_texts())
    assert legend == ["Absolute Sum (L1)", "Paraboloid (Sphere)"], legend
    fig.canvas.draw()
    figures["combined.png"] = charts.figure_png(fig)
    plt.close(fig)

    extra_files = {}
    for name, entry in computed.items():
        fig_avg = charts.plot_average_best(
            entry["per_iteration"], f"{DISPLAY_NAMES[name]}: average-best fitness")
        fig_avg.canvas.draw()
        figures[f"{name}_avg.png"] = charts.figure_png(fig_avg)
        plt.close(fig_avg)
        fig_err = charts.plot_error(entry["per_iteration"], f"{DISPLAY_NAMES[name]}: error")
        fig_err.canvas.draw()
        plt.close(fig_err)
        charts.statistics_csv(name, entry["per_iteration"])
        charts.histories_csv(name, results[name], _OPT)
        charts.runs_csv(name, results[name], _OPT)
        meta = charts.landscape_meta(name, 2, [0, 1], {}, 1, -5.0, 5.0,
                                     display_name=DISPLAY_NAMES[name])
        assert meta["display_name"] == DISPLAY_NAMES[name]
        extra_files[f"landscapes/{name}.json"] = "[]"

    trajectory = results["paraboloid"][0].trajectory
    assert trajectory is not None and trajectory.n_frames() >= 2
    func = hba_experiment.FUNCTIONS["paraboloid"]
    grid = charts.evaluate_landscape_2d(func, 2, -5.0, 5.0, 0, 1, np.zeros(2))
    anim = charts.build_animation_contour(
        *grid, trajectory, 0, 1, "paraboloid", 1, trajectory.seed, -5.0, 5.0,
        display_name=DISPLAY_NAMES["paraboloid"])
    assert DISPLAY_NAMES["paraboloid"] in anim.layout.title.text
    assert all(list(frame.traces or []) == [1, 2] for frame in anim.frames)
    extra_files["animations/paraboloid.html"] = charts.animation_html(anim)

    payload = charts.build_zip(results, computed, settings, figures,
                               extra={"master_seed_auto": False},
                               extra_files=extra_files)
    contained = sorted(zipfile.ZipFile(io.BytesIO(payload)).namelist())
    assert "settings.json" in contained and "animations/paraboloid.html" in contained
    print(f"SMOKE OK: revision 2, {len(contained)} archive entries, "
          f"{trajectory.n_frames()} animation frames")


if __name__ == "__main__":
    main()
