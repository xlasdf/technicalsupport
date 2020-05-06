#nonlinear with Polynomial

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


n = 100
x = -5 + 15*np.random.rand(n, 1)
noise = 10*np.random.randn(n, 1)
y = 10 + 1*x + 2*x**2 + noise

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
d = 3


polybasis = np.hstack([xp**i for i in range(d)])
polybasis = np.asmatrix(polybasis)

A = np.hstack([x**i for i in range(d)])
A = np.asmatrix(A)

theta = np.linalg.inv(A.T*A)*A.T*y

yp = polybasis*theta


plt.figure(figsize = (10,6))
plt.suptitle('Regression', fontsize=15)
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'Polynomial')
plt.title('Regression with Polynomial basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()
