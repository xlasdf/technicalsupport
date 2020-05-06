# Linear Basis Function Models

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

d = 6

xp = np.arange(-1, 1, 0.01).reshape(-1,1)
polybasis = np.hstack([xp**i for i in range(d)])

plt.figure(figsize = (10,8))

for i in range(d):
    plt.plot(xp, polybasis[:,i], label = '$x^{}$'.format(i))


plt.title('Polynomial Basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-1,1,-1.1,1.1])
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()
