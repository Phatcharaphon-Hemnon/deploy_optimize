import secrets

import numpy as np

import hba_experiment
from plot import Monitor
from plot import _fitness
from plot import plot_results

agents = 50
D = 3
lb = -5
up = 5
t = 20
seed = secrets.randbelow(2**32)
# Single HBA implementation: the legacy demonstration runs one repetition of
# the corrected optimizer on the Griewank objective (``plot._fitness``) and
# reuses its recorded trajectory for the per-iteration frames.
result = hba_experiment.optimize(
    "griewank", D, agents, t, lb, up, seed,
    record_frames=list(range(t + 1)), record_repetition=1,
)
monitor = Monitor(lb, up)
for k, iteration in enumerate(result.trajectory.frame_iterations):
    if int(iteration) == 0:
        continue  # initialization frame has no update yet; video shows updates
    monitor.save_frame(
        np.asarray(result.trajectory.populations[k], dtype=float),
        np.asarray(result.trajectory.best_positions[k], dtype=float),
        int(iteration),
    )

print(result.best_fitness)
print(result.best_position)
monitor.build_video(t)
plot_results(
    np.asarray(result.trajectory.populations[-1], dtype=float),
    np.asarray(result.best_position, dtype=float),
    lb, up,
)
