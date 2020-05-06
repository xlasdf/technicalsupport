# nonlinear with RBF Function

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


n = 100
x = -5 + 15*np.random.rand(n, 1)
noise = 10*np.random.randn(n, 1)
y = 10 + 1*x + 2*x**2 + noise

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)

d = 6
u = np.linspace(np.min(x), np.max(x), d)
sigma = 5

rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.asmatrix(rbfbasis)

A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
A = np.asmatrix(A)

theta = np.linalg.inv(A.T*A)*A.T*y

yp = rbfbasis*theta

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'RBF')
plt.title('Regression with RBF basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()
