import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

H = np.array([[2,0],[0,2]])
f = -np.array([[6],[-6]])

x = np.zeros((2,1))

alpha = 0.2

for i in range(25):
    g = H.dot(x) + f
    x = x - alpha*g

print(x)
