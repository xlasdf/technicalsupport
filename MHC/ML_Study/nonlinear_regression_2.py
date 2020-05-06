import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

n = 100
x = -5 + 15*np.random.rand(n, 1)
noise = 10*np.random.randn(n, 1)
y = 10 + 1*x + 2*x**2 + noise

A = np.hstack([x**0])
