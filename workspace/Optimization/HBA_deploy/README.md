# HBA comparison

Honey Badger Algorithm (HBA) experiments on six local minimization objectives,
with a Streamlit dashboard comparing convergence across independent repetitions.

## Streamlit app (new)

Interactive entrypoint: `streamlit_app.py`.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Defaults: **All functions** (or pick one of the six), 50 agents, 3 dimensions,
200 iterations, 30 repetitions, shared bounds `[-5, 5]`, and a blank
**Master seed (optional)**. All sizes are configurable in the form (positive
integers within the stated caps).
Counts must be positive; dimensions are configurable with Rosenbrock requiring
at least 2. Bounds must be finite and contain a known minimizer for every
selected function: coordinate 1 for Rosenbrock and 0 for the others. Very large
experiments require correspondingly more compute and memory. Objective or update
overflow produces an error rather than invalid data.

The form's **Run experiment** button runs the optimization with a progress
indicator. A blank master seed is resolved to a random valid integer only when
the button is submitted; an explicit seed (including `0`) stays deterministic.
The resolved seed is shown with the completed results and retained in settings
and downloads: re-entering it reproduces the run in the same software
environment. Completed results and their settings stay in session state, so
changing chart, landscape, or download options never reruns optimization and
never generates new seeds.
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
  or symmetric-log axes that preserve zero, per-repetition best positions, the
  exact local formula, and an optional objective-landscape view (off by
  default): **2D contour**, **3D surface**, or **Both**. Landscapes evaluate the
  selected function on a fixed `100 × 100` grid within the experiment's bounds —
  the full landscape for 2D experiments, a one-dimensional line plot for 1D
  experiments, and a two-coordinate slice for higher dimensions (remaining
  coordinates held at a selected repetition's best position, default repetition
  1). Surface height is objective fitness, not a third optimization coordinate.
  Constant landscapes are annotated; grids with no finite values show a message
  instead of a plot. Generated landscape images have individual PNG downloads
  and are included in the ZIP with slice metadata.
- **Downloads**: per-function statistics/histories/best-position CSVs,
  settings/seeds JSON, PNG charts, generated landscape images, and a ZIP
  archive. Files are generated in memory per session; nothing is written to
  disk.

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

- `rosenbrock` (Rosenbrock): sum of `100 * (x[i]**2 - x[i+1])**2 + (1 - x[i])**2` for adjacent coordinates.
- `powell_sum` (Sum of Different Powers): sum of `abs(x[i])**(i + 2)` — exponents run `2..D+1` (`i` starts at 0).
- `schwefel` (Absolute Sum (L1)): sum of `abs(x[i])` — an absolute sum, not the commonly named Schwefel 2.26 objective.
- `paraboloid` (Paraboloid (Sphere)): sum of `x[i]**2` over every coordinate.
- `rastrigin` (Rastrigin): `10*D + sum(x[i]**2 - 10*cos(2*pi*x[i]))`.
- `griewank` (Griewank): `1 + sum(x[i]**2)/4000 - prod(cos(x[i] / sqrt(i + 1)))`.

Display labels in charts, legends, and export metadata use the parenthesized
names; internal identifiers (code keys, CSV/JSON fields, archive paths) are
unchanged. Results carry a model revision (`MODEL_REVISION` in
`hba_experiment`); the dashboard clears results retained from older revisions
instead of mixing them with new runs.

All comparisons are results for these local implementations. HBA uses the
standard update equations implemented once in `hba_experiment.optimize`:
uniform initialization with one fitness evaluation per agent, an independent
best-position/fitness copy, density factor `alpha = 2*exp(-t/T)` for
`t = 1..T` with `beta = 6`, per-iteration intensities from a population
snapshot (`r*S / (4*pi*(d + 1e-10)**2)` with squared Euclidean neighbor
distances and circular indexing), per-agent digging/honey choice with equal
probability, one sign `F` from `{-1, +1}`, independent per-coordinate random
coefficients, clipping to bounds, one evaluation per candidate, and greedy
acceptance (a candidate no worse than the agent's fitness is kept, updating
the global-best copy immediately). History index 0 is the initialization.
Same seeds reproduce a run in the same software environment; results differ
from the pre-correction implementation.

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
statistics, Matplotlib convergence figures, dimension-aware landscape
evaluation/plotting helpers, and in-memory CSV/JSON/PNG/ZIP exports (no new
plotting dependencies).

## Existing scripts

`_tool.py` and `plot.py` are left as-is.

- `main.py` runs a fixed HBA demonstration: 50 agents, 3 dimensions, bounds
  `[-5, 5]`, 20 iterations on the Griewank objective (`plot._fitness`) by
  calling `hba_experiment.optimize` (the single implementation) with a random
  seed and a full trajectory recording. It saves per-iteration frames to
  `frames/`, builds an `.mp4` video, saves a 3D surface/contour figure, and
  prints the best fitness and position.
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
