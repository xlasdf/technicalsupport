import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
#https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/05_Regression_1-1.html
x = np.array([0.2, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1,1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1,1)
# print(x,'\n')
# print(y,'\n')

# add outliers
x = np.vstack([x, np.array([0.5, 3.8]).reshape(-1,1)])
y = np.vstack([y, np.array([3.9, 0.3]).reshape(-1,1)])
# print(x,'\n')
# print(y,'\n')

# A = np.hstack([x, np.ones([x.shape[0],1])])

# m = y.shape[0]
# A = np.hstack([np.ones([m,1]), x])
A = np.hstack([x**0,x])
A = np.asmatrix(A)

theta1 = cvx.Variable([2,1])
obj1 = cvx.Minimize(cvx.norm(A*theta1-y,1))
ans = cvx.Problem(obj1).solve()

theta2 = cvx.Variable([2,1])
obj2 = cvx.Minimize(cvx.norm(A*theta2-y,2))
ans2 = cvx.Problem(obj2).solve()

# to plot data

plt.figure(figsize=(10,8))
plt.plot(x, y, 'ko',label='data')
plt.title('$L_1$ and $L_2$ Regression w/ Outliers', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)

# to plot straight lines (fitted lines)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp1 = theta1.value[0,0] + theta1.value[1,0]*xp
yp2 = theta2.value[0,0] + theta2.value[1,0]*xp   # y = x1*theta + theta2

plt.plot(xp, yp1, 'b', linewidth = 2, label = '$L_1$')
plt.plot(xp, yp2, 'r', linewidth = 2, label = '$L_2$')
plt.axis('scaled')
plt.xlim([0, 5])
plt.legend(fontsize = 15, loc = 5)
plt.grid(alpha = 0.3)
plt.show()
