"""Configurable Honey Badger Algorithm (HBA) experiments.

This module reuses the exact update equations from ``main.py`` (digging /
honey phases, intensity, density factor ``alpha``, greedy acceptance) without
importing ``main.py`` itself (which executes an experiment on import).

Objective functions are imported directly from ``_tool.py`` exactly as
implemented there; they are intentionally not corrected here. In particular:

- ``paraboloid`` overwrites its accumulator, so it uses only the last
  coordinate (``x[-1]**2``).
- ``powell_sum`` computes weighted squares ``abs(i * x[i]**2)``.
- ``schwefel`` computes an absolute sum ``sum(abs(x[i]))``, not the
  commonly named Schwefel 2.26 objective.

Comparisons produced with these callables are results for these local
implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from _tool import _DistanceBetween
from _tool import griewank
from _tool import paraboloid
from _tool import powell_sum
from _tool import rastrigin
from _tool import rosenbrock
from _tool import schwefel

# ---------------------------------------------------------------------------
# Registry and constants
# ---------------------------------------------------------------------------

FUNCTIONS = {
    "rosenbrock": rosenbrock,
    "powell_sum": powell_sum,
    "schwefel": schwefel,
    "paraboloid": paraboloid,
    "rastrigin": rastrigin,
    "griewank": griewank,
}

FUNCTION_ORDER = ["rosenbrock", "powell_sum", "schwefel", "paraboloid", "rastrigin", "griewank"]

#: All six local implementations attain 0 at a known minimizer when the
#: bounds contain it (1-vector for Rosenbrock, 0-vector for the others).
KNOWN_OPTIMUM = 0.0

#: Displayed formulas (exact local behaviour, not textbook definitions).
FORMULAS = {
    "rosenbrock": "sum of 100*(x[i]**2 - x[i+1])**2 + (1 - x[i])**2 for adjacent coordinates",
    "powell_sum": "sum of abs(i * (x[i]**2))  [weighted squares; i starts at 0, so the first term is always 0]",
    "schwefel": "sum of abs(x[i])  [absolute sum; not the commonly named Schwefel 2.26 objective]",
    "paraboloid": "x[-1]**2  [loop overwrites the accumulator, so only the last coordinate is used]",
    "rastrigin": "10*D + sum(x[i]**2 - 10*cos(2*pi*x[i]))",
    "griewank": "1 + sum(x[i]**2)/4000 - prod(cos(x[i] / sqrt(i + 1)))",
}

MINIMIZER_NOTES = {
    "rosenbrock": "Known minimizer (1, ..., 1) with value 0; requires 1 in bounds and at least 2 dimensions.",
    "powell_sum": "Known minimizer (0, ..., 0) with value 0; requires 0 in bounds.",
    "schwefel": "Known minimizer (0, ..., 0) with value 0; requires 0 in bounds.",
    "paraboloid": "A known minimizer is (0, ..., 0) with value 0 (any vector with last coordinate 0 attains 0); requires 0 in bounds.",
    "rastrigin": "Known minimizer (0, ..., 0) with value 0; requires 0 in bounds.",
    "griewank": "Known minimizer (0, ..., 0) with value 0; requires 0 in bounds.",
}

DEFAULTS = {
    "functions": list(FUNCTION_ORDER),
    "agents": 50,
    "dimensions": 3,
    "iterations": 200,
    "repetitions": 30,
    "lb": -5.0,
    "ub": 5.0,
    "seed": 42,
}

#: HBA constants matching ``main.py``.
_HBA_C = 1.0
_HBA_BETA = 6.0
_HBA_EPS = 1e-10

#: Safety caps so a dashboard run stays within available resources.
LIMITS = {
    "agents": 200,
    "dimensions": 50,
    "iterations": 2000,
    "repetitions": 100,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_positive_int(name, value, limit=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    if limit is not None and value > limit:
        raise ValueError(f"{name} must be <= {limit} for available resources, got {value}.")
    return value


def validate_counts(agents, dimensions, iterations, repetitions, functions=None):
    """Validate experiment sizes; returns validated ``(agents, dim, iters, reps)``."""
    functions = list(functions) if functions is not None else list(FUNCTION_ORDER)
    agents = _require_positive_int("agents", agents, LIMITS["agents"])
    dimensions = _require_positive_int("dimensions", dimensions, LIMITS["dimensions"])
    iterations = _require_positive_int("iterations", iterations, LIMITS["iterations"])
    repetitions = _require_positive_int("repetitions", repetitions, LIMITS["repetitions"])
    if "rosenbrock" in functions and dimensions < 2:
        raise ValueError("Rosenbrock requires at least 2 dimensions.")
    return agents, dimensions, iterations, repetitions


def validate_bounds(lb, ub, functions):
    """Validate shared bounds; they must contain each selected minimizer."""
    functions = list(functions)
    try:
        lb_f = float(lb)
        ub_f = float(ub)
    except (TypeError, ValueError):
        raise ValueError(f"Bounds must be numeric, got lb={lb!r}, ub={ub!r}.")
    if not (math.isfinite(lb_f) and math.isfinite(ub_f)):
        raise ValueError("Bounds must be finite.")
    if not lb_f < ub_f:
        raise ValueError(f"Lower bound must be below upper bound, got [{lb_f}, {ub_f}].")
    needs_zero = any(name != "rosenbrock" for name in functions)
    needs_one = "rosenbrock" in functions
    if needs_zero and not (lb_f <= 0.0 <= ub_f):
        raise ValueError("Bounds must contain coordinate 0 for the selected non-Rosenbrock functions.")
    if needs_one and not (lb_f <= 1.0 <= ub_f):
        raise ValueError("Bounds must contain coordinate 1 for Rosenbrock.")
    return lb_f, ub_f


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"Seed must be an integer, got {seed!r}.")
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed must be in [0, 2**32 - 1], got {seed}.")
    return seed


def validate_functions(names):
    if not names:
        raise ValueError("Select at least one function.")
    unknown = [n for n in names if n not in FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown functions: {unknown}.")
    # Preserve canonical order.
    return [n for n in FUNCTION_ORDER if n in names]


def _check_finite_vector(vec, context):
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{context} must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} produced a non-finite position.")
    return arr


def _evaluate(func, position):
    try:
        value = float(func(np.asarray(position, dtype=float)))
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Objective evaluation failed: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError("Objective or update overflow produced a non-finite fitness.")
    return value


def derive_seeds(master_seed, repetitions):
    """Return one independent seed per repetition.

    The schedule depends only on the master seed and repetition index, so it
    is identical across functions: repetition ``r`` of every function uses
    ``seeds[r]``.
    """
    master_seed = validate_seed(master_seed)
    repetitions = _require_positive_int("repetitions", repetitions, LIMITS["repetitions"])
    return [master_seed + r for r in range(repetitions)]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    function_name: str
    seed: int
    best_position: np.ndarray
    best_fitness: float
    best_history: np.ndarray  # length iterations + 1, includes initialization
    dimensions: int = 0
    agents: int = 0
    iterations: int = 0

    def __post_init__(self):
        self.best_position = np.asarray(self.best_position, dtype=float)
        self.best_history = np.asarray(self.best_history, dtype=float)
        if self.best_position.ndim != 1:
            raise ValueError("best_position must be one-dimensional.")
        if self.best_position.shape[0] != self.dimensions and self.dimensions:
            raise ValueError("best_position length must match dimensions.")
        if self.best_history.ndim != 1:
            raise ValueError("best_history must be one-dimensional.")
        if self.best_history.shape[0] != self.iterations + 1 and self.iterations:
            raise ValueError("best_history must include initialization (iterations + 1 values).")
        if not np.all(np.isfinite(self.best_history)):
            raise ValueError("best_history must be finite.")
        if not math.isfinite(float(self.best_fitness)):
            raise ValueError("best_fitness must be finite.")


@dataclass
class ExperimentSettings:
    functions: list = field(default_factory=lambda: list(FUNCTION_ORDER))
    agents: int = DEFAULTS["agents"]
    dimensions: int = DEFAULTS["dimensions"]
    iterations: int = DEFAULTS["iterations"]
    repetitions: int = DEFAULTS["repetitions"]
    lb: float = DEFAULTS["lb"]
    ub: float = DEFAULTS["ub"]
    seed: int = DEFAULTS["seed"]
    seeds: list = field(default_factory=list)

    def to_dict(self):
        return {
            "functions": list(self.functions),
            "agents": self.agents,
            "dimensions": self.dimensions,
            "iterations": self.iterations,
            "repetitions": self.repetitions,
            "lb": self.lb,
            "ub": self.ub,
            "seed": self.seed,
            "seeds": list(self.seeds),
            "known_optimum": KNOWN_OPTIMUM,
        }


# ---------------------------------------------------------------------------
# Optimizer (equations matching ``main.py``)
# ---------------------------------------------------------------------------


def optimize(function_name, dimensions, agents, iterations, lb, ub, seed):
    """Run one HBA repetition; return a :class:`RunResult`.

    Uses the update equations from ``main.py`` with ``c=1``, ``beta=6`` and
    ``eps=1e-10``: uniform initialization scaled to ``[lb, ub]``, proper
    best-population initialization (argmin fitness), per-agent digging/honey
    updates with clipping to bounds, and greedy acceptance for both the agent
    and the global best (``<=`` comparisons, as in ``main.py``).
    """
    functions = validate_functions([function_name])
    agents, dimensions, iterations, _reps = validate_counts(agents, dimensions, iterations, 1, functions)
    lb_f, ub_f = validate_bounds(lb, ub, functions)
    seed = validate_seed(seed)
    func = FUNCTIONS[function_name]

    rng = np.random.default_rng(seed)
    pop = rng.uniform(low=0.0, high=1.0, size=(agents, dimensions)) * (ub_f - lb_f) + lb_f
    pop = np.clip(pop, lb_f, ub_f)
    fitness = np.empty(agents, dtype=float)
    for i in range(agents):
        fitness[i] = _evaluate(func, pop[i])
    best_idx = int(np.argmin(fitness))
    best_fitness = float(fitness[best_idx])
    history = [best_fitness]

    for k in range(iterations):
        alpha = _HBA_C * math.exp(-k / iterations)
        for i in range(agents):
            r3 = float(rng.uniform(low=0.0, high=1.0))
            r4 = float(rng.uniform(low=0.0, high=1.0))
            r5 = float(rng.uniform(low=0.0, high=1.0))
            r7 = float(rng.uniform(low=0.0, high=1.0))
            f_draw = float(rng.uniform(low=0.0, high=1.0))
            flag = 1 if f_draw >= 0.5 else -1
            coin = float(rng.uniform(low=0.0, high=1.0))
            dist_next = _DistanceBetween(pop[i], pop[(i + 1) % agents])
            if dist_next is None:
                raise ValueError("Position dimensionality mismatch in distance computation.")
            dt2 = float(dist_next) ** 2
            dist_best = _DistanceBetween(pop[best_idx], pop[i])
            if dist_best is None:
                raise ValueError("Position dimensionality mismatch in distance computation.")
            dt = float(dist_best) + _HBA_EPS
            intensity_draw = float(rng.uniform(low=0.0, high=1.0))
            intensity = intensity_draw * dt2 / (4.0 * math.pi * (dt**2))
            best_pos = pop[best_idx]
            if coin < 0.5:
                shape = math.cos(2.0 * math.pi * r4) * (1.0 - math.cos(2.0 * math.pi * r5))
                new_pos = best_pos + (flag * _HBA_BETA * intensity) * best_pos + (
                    flag * r3 * alpha * dt * shape
                )
            else:
                new_pos = best_pos + flag * r7 * alpha * dt
            new_pos = np.clip(np.asarray(new_pos, dtype=float), lb_f, ub_f)
            new_pos = _check_finite_vector(new_pos, "Updated position")
            if new_pos.shape[0] != dimensions:
                raise ValueError("Updated position dimensionality mismatch.")
            new_fit = _evaluate(func, new_pos)
            old_best = float(fitness[best_idx])
            if new_fit <= float(fitness[i]):
                pop[i] = new_pos
                fitness[i] = new_fit
            if new_fit <= old_best:
                # ``<=`` moves the best index on ties, matching ``main.py``.
                best_idx = i
        history.append(float(np.min(fitness)))

    best_idx = int(np.argmin(fitness))
    best_fitness = float(fitness[best_idx])
    best_position = np.array(pop[best_idx], dtype=float)
    return RunResult(
        function_name=function_name,
        seed=seed,
        best_position=best_position,
        best_fitness=best_fitness,
        best_history=np.asarray(history, dtype=float),
        dimensions=dimensions,
        agents=agents,
        iterations=iterations,
    )


def run_experiment(function_names, agents=DEFAULTS["agents"], dimensions=DEFAULTS["dimensions"],
                   iterations=DEFAULTS["iterations"], repetitions=DEFAULTS["repetitions"],
                   lb=DEFAULTS["lb"], ub=DEFAULTS["ub"], seed=DEFAULTS["seed"],
                   progress_callback=None):
    """Run every selected function for all repetitions.

    Repetition ``r`` of every function uses seed ``seeds[r]`` (same schedule
    across functions). Returns ``(results, settings)`` where ``results`` maps
    function name to a list of :class:`RunResult`.
    """
    function_names = validate_functions(list(function_names))
    agents, dimensions, iterations, repetitions = validate_counts(
        agents, dimensions, iterations, repetitions, function_names
    )
    lb_f, ub_f = validate_bounds(lb, ub, function_names)
    master = validate_seed(seed)
    seeds = derive_seeds(master, repetitions)
    settings = ExperimentSettings(
        functions=function_names, agents=agents, dimensions=dimensions,
        iterations=iterations, repetitions=repetitions, lb=lb_f, ub=ub_f,
        seed=master, seeds=seeds,
    )
    results = {}
    total = len(function_names) * repetitions
    done = 0
    for name in function_names:
        runs = []
        for r in range(repetitions):
            runs.append(
                optimize(name, dimensions, agents, iterations, lb_f, ub_f, seeds[r])
            )
            done += 1
            if progress_callback is not None:
                progress_callback(done / total)
        results[name] = runs
    return results, settings
