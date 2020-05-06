import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

m = 100
x1 = 8*np.random.rand(m,1)
x2 = 7*np.random.rand(m,1) - 4

g = 0.8*x1 + x2 -3

C1 = np.where(g >= 0)[0]
C0 = np.where(g <= 0)[0]

y = np.empty([m,1])

y[C1] = 1
y[C0] = 0

X = np.hstack([np.ones([m,1]), x1, x2])
X = np.asmatrix(X)
w = cvx.Variable([3,1])

obj = cvx.Maximize(y.T*X*w - cvx.sum(cvx.logistic(X*w)))
prob = cvx.Problem(obj).solve()

w = w.value
xp = np.linspace(-1,9,100).reshape(-1, 1)
yp = -w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize=(10,8))
plt.plot(X[C1,1], X[C1,2], 'ro', alpha=0.3, label='C1')
plt.plot(X[C0,1], X[C0,2], 'bo', alpha=0.3, label='C0')
plt.plot(xp, yp, 'g', linewidth=4, label='Logistic Regression')
plt.title('Logistic Regression', fontsize=15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0,9])
plt.ylim([-4, 3])
plt.show()
