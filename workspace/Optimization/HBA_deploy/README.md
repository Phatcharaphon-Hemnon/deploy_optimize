# HBA comparison

Honey Badger Algorithm (HBA) experiments on six local minimization objectives,
with a Streamlit dashboard comparing convergence across independent repetitions.

## Streamlit app (new)

Interactive entrypoint: `streamlit_app.py`.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Defaults: all six functions, 50 agents, 3 dimensions, 200 iterations,
30 repetitions, shared bounds `[-5, 5]`, master seed `42`. All sizes are
configurable in the form (positive integers within the stated caps).
Counts must be positive; dimensions are configurable with Rosenbrock requiring
at least 2. Bounds must be finite and contain a known minimizer for every
selected function: coordinate 1 for Rosenbrock and 0 for the others. Very large
experiments require correspondingly more compute and memory. Objective or update
overflow produces an error rather than invalid data.

The form's **Run experiment** button runs the optimization with a progress
indicator. Completed results and their settings stay in session state, so
changing chart options or downloading files never reruns optimization.
Repetitions use independent derived seeds (`master + repetition index`); the
same seed schedule is used across functions, so repeating the settings and
master seed reproduces numerical results in the same software environment.

Tabs:

- **Comparison**: combined mean-error chart plus a sortable table of final
  mean, median, sample standard deviation, minimum, and maximum fitness/error
  per function.
- **Function details**: average-best chart (mean and median best-so-far fitness
  with a mean ± one sample-standard-deviation band), error chart
  (`abs(best_fitness - known_optimum)` with mean, median, and band) on linear
  or symmetric-log axes that preserve zero, per-repetition best positions, and
  the exact local formula.
- **Downloads**: per-function statistics/histories/best-position CSVs,
  settings/seeds JSON, PNG charts, and a ZIP archive. Files are generated in
  memory per session; nothing is written to disk.

With one repetition, variability bands are omitted and standard deviation is
shown as unavailable.

## Reading the results

**Average-best** is the mean of the runs' best-so-far fitness at each iteration,
not the average fitness of agents within a population. Iteration 0 records the
best initialized agent. **Error** is `abs(best_fitness - known_optimum)`.
All six local functions have optimum value 0 when the bounds contain a known
minimizer, so fitness and error statistics coincide for these nonnegative
functions; the error chart uses a different axis scale to show convergence near
zero.

Shaded bands are mean ± one **sample standard deviation** (`ddof=1`), not
confidence intervals. The lower error band is clipped to zero for display only.
With one run, standard deviation is unavailable and the band is omitted. Error
charts offer a symmetric-log axis with a linear region at errors below `1e-12`,
preserving exact zeros without altering the data.

Function scales differ, so lower raw errors across different functions do not
establish an overall ranking. Shared bounds are an experimental choice, not a
claim that these are each benchmark's conventional bounds.

## Objective definitions (local implementations)

For `i = 0, ..., D - 1`, exactly as implemented in `_tool.py`:

- `rosenbrock`: sum of `100 * (x[i]**2 - x[i+1])**2 + (1 - x[i])**2` for adjacent coordinates.
- `powell_sum`: sum of `abs(i * (x[i]**2))` — weighted squares (`i` starts at 0, so the first term is always 0).
- `schwefel`: sum of `abs(x[i])` — an absolute sum, not the commonly named Schwefel 2.26 objective.
- `paraboloid`: `x[-1]**2` — the loop overwrites its accumulator, so only the last coordinate is used.
- `rastrigin`: `10*D + sum(x[i]**2 - 10*cos(2*pi*x[i]))`.
- `griewank`: `1 + sum(x[i]**2)/4000 - prod(cos(x[i] / sqrt(i + 1)))`.

All comparisons are results for these local implementations. HBA reuses the
update equations and greedy acceptance from `main.py`, with proper
best-population initialization.

## Python API

```python
from hba_experiment import optimize, run_experiment

result = optimize("paraboloid", dimensions=3, agents=50, iterations=200,
                  lb=-5.0, ub=5.0, seed=42)
print(result.best_fitness, result.best_position)
# result.best_history has iterations + 1 values, including initialization.

results, settings = run_experiment(
    ["rosenbrock", "griewank"], agents=50, dimensions=3,
    iterations=200, repetitions=30, lb=-5.0, ub=5.0, seed=42)
```

`hba_experiment` imports objectives directly from `_tool.py`. It never imports
`main.py`, which executes an experiment on import. `charts.py` provides
statistics, Matplotlib figures, and in-memory CSV/JSON/PNG/ZIP exports.

## Existing scripts (unchanged)

`_tool.py`, `main.py`, and `plot.py` are left as-is.

- `main.py` runs a fixed HBA demonstration: 50 agents, 3 dimensions, bounds
  `[-5, 5]`, 20 iterations on the Griewank objective (`plot._fitness`), with
  greedy acceptance. It saves per-iteration frames to `frames/`, builds an
  `.mp4` video, saves a 3D surface/contour figure, and prints the best fitness
  and position.
- `plot.py` provides `plot_results` (3D surface + contour) and the `Monitor`
  frame/video helper used by `main.py`.
- `_tool.py` defines the six objectives above plus the `_DistanceBetween`
  helper.

```bash
.venv/bin/python main.py
```

The animation/video path needs OpenCV (`pip install opencv-python`); it is
intentionally not part of the dashboard's dependencies (`requirements.txt`).

## Deployment (Streamlit Community Cloud)

The app is ready to deploy; publishing the live app remains a separate action.

1. Push this repository to GitHub (branch `main` or your chosen branch).
2. At [share.streamlit.io](https://share.streamlit.io): **New app** → select the
   repository, branch, and entrypoint `streamlit_app.py` (file in the repo root).
3. Keep the default Python environment; dependencies install from
   `requirements.txt` (`numpy`, `matplotlib`, `streamlit`). No system packages,
   secrets, or OpenCV are required.
4. Deploy, then open the app URL and press **Run experiment** to verify charts
   and downloads.

Local startup remains `streamlit run streamlit_app.py`.
