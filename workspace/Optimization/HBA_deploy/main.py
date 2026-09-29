import numpy as np
import random as rd
import math as m
from plot import plot_results
from plot import Monitor
from _tool import _DistanceBetween
from plot import _fitness
agents = 50 
D = 3
lb = -5
up = 5
rng = np.random.default_rng()
arr = rng.uniform(low = 0, high = 1.0, size=(agents, D)) * (up-lb) + lb
_best = 0
_ybest = _fitness(arr[_best])
t = 20
c = 1
_beta = 6
eps = 1e-10
monitor = Monitor(lb, up)
for k in range(t):
    _alpha = c*(m.exp(-k/t))
    y = np.ones((agents, D))
    for i in range(len(arr)):
        r3 = rng.uniform(low = 0.0 , high = 1.0)
        r4 = rng.uniform(low = 0.0 , high = 1.0)
        r5 = rng.uniform(low = 0.0 , high = 1.0)
        r7 = rng.uniform(low = 0.0 , high = 1.0)
        F = 1 if rng.uniform(low=0.0, high = 1.0) >= 0.5 else -1
        _dt2 = _DistanceBetween(arr[i], arr[(i+1)%agents])**2
        dt = _DistanceBetween(arr[_best], arr[i]) + eps
        _intensity = rng.uniform(low = 0.0, high = 1.0) * _dt2 / (4*(m.pi)*(dt**2))
        if rng.uniform(low = 0.0 , high = 1.0) < 0.5:
            _newpos = arr[_best] + (F * _beta * _intensity) * arr[_best] + (F * r3 * _alpha * dt * (m.cos(2 * m.pi * r4) * (1-m.cos(2 * m.pi * r5))))
        else:
            _newpos = arr[_best] + F * r7 * _alpha * dt
        _newpos = np.clip(_newpos, lb, up)
        if (_fitness(_newpos) <= _fitness(arr[i])):
            arr[i] = _newpos
            y[i] = _fitness(_newpos)
        if (_fitness(_newpos) <= _fitness(arr[_best])):
            _best = i
            _ybest = _fitness(arr[_best])
    monitor.save_frame(arr, arr[_best], k)

print(_ybest)
print(arr[_best])
monitor.build_video(t)
plot_results(arr, arr[_best], lb, up)
