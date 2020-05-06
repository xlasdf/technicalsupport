import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

x = np.array([0.2, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1,1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1,1)

m = y.shape[0]

A = np.hstack([np.ones([m,1]), x])

A = np.asmatrix(A)

#2 norm
theta2 = cvx.Variable([2,1])

obj = cvx.Minimize(cvx.norm(A*theta2-y, 2)) #A*theta2 = y^, 2norm 쓰겠다.
cvx.Problem(obj,[]).solve()


print('theta:\n', theta2.value)

# 1norm
theta1 = cvx.Variable([2,1])
obj1 = cvx.Minimize(cvx.norm(A*theta1-y, 1)) # 1norm 쓰겠다.
cvx.Problem(obj1).solve()

print('theta:\n', theta1.value)
# to plot data

plt.figure(figsize=(10,6))
plt.title('$L_1$ and $L_2$ Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko',label='data')

# to plot straight lines (fitted lines)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp1 = theta1.value[0,0]*xp + theta1.value[1,0]
yp2 = theta2.value[0,0]*xp + theta2.value[1,0]   # y = x1*theta + theta2

plt.plot(xp, yp1, 'b', linewidth=2, label='$L_1$')
plt.plot(xp, yp2, 'r', linewidth=2, label='$L_2$')
plt.legend(fontsize=15)
plt.axis('scaled')
plt.xlim([0,5])
plt.grid(alpha=0.3)
plt.show()
