import numpy as np
import math as m
def _DistanceBetween(arr1, arr2):
    dt2 = 0
    if len(arr1) != len(arr2):
        return
    for i in range(len(arr1)):
        dt2 += (arr1[i] - arr2[i])**2
    return m.sqrt(dt2)   

def rosenbrock(x):
    total = 0
    for i in range(len(x)-1):
        total += 100*(x[i]**2 - x[i+1])**2 + (1-x[i])**2
    return total

def powell_sum(x):
    total = 0
    for i in range(len(x)):
        total += abs(x[i]) ** (i + 2)
    return total

def schwefel(x):
    total = 0
    for i in range(len(x)):
        total += abs(x[i])
    return total
def paraboloid(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total
def rastrigin(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2 - 10*m.cos(2*m.pi*x[i])
    return 10*len(x) + total
def griewank(x):
    _sum = 0
    _prod = 1 
    for i in range(len(x)):
        _sum += x[i]**2
        _prod *= m.cos(x[i] / m.sqrt(i + 1))
        
    return 1 + (1/4000) * _sum - _prod
