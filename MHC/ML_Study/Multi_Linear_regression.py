import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

# for 3D plot
from mpl_toolkits.mplot3d import Axes3D


# y = theta1*x1 + theta2*x2 + theta3 + noise


n = 200
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
noise = 0.5*np.random.randn(n, 1)
y = 1*x1 + 3*x2 +2 + noise

A = np.hstack([x1, x2, np.ones((n, 1))])
A = np.asmatrix(A)
theta = np.linalg.inv(A.T*A)*A.T*y
print(theta,'\n')
X1, X2 = np.meshgrid(np.arange(np.min(x1),np.max(x1),0.5), np.arange(np.min(x2), np.max(x2),0.5))
print(X1,'\n')
YP = theta[0,0]*X1 + theta[1,0]*X2 + theta[2,0]



# print(x1,'\n')
# print(x2,'\n')
# print(noise, '\n')



fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('Generated Data',fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', label='Data')
ax.plot_wireframe(X1,X2,YP,color='k', alpha=0.3, label='Regression Plane')
ax.view_init(90,0) # 각도 조정
plt.legend(fontsize=15)
plt.show()
