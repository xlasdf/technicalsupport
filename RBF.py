# RBF Function
# Radial Basis Functions 
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

# 갯수와 sigma를 가지고 control 가능하다.
# sigma가 작을수록 서로간의 영향 줄어듬.
d = 9

xp = np.arange(-1, 1, 0.01).reshape(-1,1)
u = np.linspace(-1, 1, d)
sigma = 0.3

rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
plt.figure(figsize = (10,8))

for i in range(d):
    plt.plot(xp, rbfbasis[:,i], label = '$\mu = {}$'.format(u[i]))

plt.title('RBF basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-1, 1, -0.1, 1.1])
plt.grid(alpha = 0.3)
plt.legend(loc = 'lower right', fontsize=15)
plt.show()
