# nonlinear Regression

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

n = 100
x = -5 + 15*np.random.rand(n, 1)
noise = 10*np.random.randn(n, 1)
y = 10 + 1*x + 2*x**2 + noise

A = np.hstack([np.ones((n,1)), x, x**2])
A = np.asmatrix(A)

theta = np.linalg.inv(A.T*A)*A.T*y

xp = np.linspace(np.min(x),np.max(x))
yp = theta[0,0] + theta[1,0]*xp + theta[2,0]*xp**2



plt.figure(figsize=(10,6))
plt.title('True x and y', fontsize=15)
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.plot(x, y, 'o', markersize=4, label='actual')
plt.plot(xp, yp, 'r', linewidth=2, label='estimated')

plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()
