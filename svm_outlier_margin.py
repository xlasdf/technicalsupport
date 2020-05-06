import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
# m = 100 # 100개의 data point

x1 = 8*np.random.rand(100,1)
x2 = 7*np.random.rand(100,1) - 4


g = 0.8*x1 + x2 -3

g1 = g - 1 #g에서 -1만큼 떨어진
g0 = g + 1 #g에서 1만큼 떨어진

C1 = np.where(g1 >= 0)[0]
C0 = np.where(g0 <= 0)[0]

xp = np.linspace(-1,9,100).reshape(-1, 1)
ypt = -0.8*xp + 3

N = C1.shape[0]
M = C0.shape[0]

X1 = np.hstack([np.ones([N,1]),x1[C1], x2[C1]])
X0 = np.hstack([np.ones([M,1]),x1[C0], x2[C0]])

outlier1 = np.array([1, 4.5, 1]) # (1, 4.5, 1)
outlier2 = np.array([1, 2, 2]) # (1, 2, 2)

X0 = np.vstack([X0, outlier1, outlier2]) # outlier 추가, 2점
X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

# 해없음, linear Separable 에서 outlier 고려 안된 알고리즘, relax 시켜줘야함!
# w = cvx.Variable(3,1)
# obj = cvx.Minimize(1)
# const = [X1*w >= 1 , X0*w <= -1]
# prob = cvx.Problem(obj, const).solve()
#
# print(w.value)

# slack Variable 적용
N = X1.shape[0] # 변수가 추가됐으므로...
M = X0.shape[0]

### margin 적용(||w||2 + gamma(1*u + 1*v)), svm model 1 ###

# g = 1 # 클수록 mis 허용 x , 작을 수록 robust
#
# w = cvx.Variable([3,1])
# u = cvx.Variable([N,1])
# v = cvx.Variable([M,1])
#
#
# obj = cvx.Minimize(cvx.norm(w,2) + g*(np.ones((1,N))*u + np.ones((1,M))*v))
# const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0]
# prob = cvx.Problem(obj, const).solve(solver = 'ECOS')
#
# w = w.value
#
# xp = np.linspace(-1,9,100).reshape(-1,1)
# yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]
#
# plt.figure(figsize = (10, 8))
# plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
# plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
# plt.plot(xp, ypt, 'k', alpha = 0.1, label = 'True')
# plt.plot(xp, ypt-1, '--k', alpha = 0.1)
# plt.plot(xp, ypt+1, '--k', alpha = 0.1)
# plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 2') # outlier relax 한 선
# plt.plot(xp, yp-1/w[2,0], '--g') # relax 선으로 부터 -1
# plt.plot(xp, yp+1/w[2,0], '--g') # relax 선으로 부터 1
# plt.title('When Outliers Exist', fontsize = 15)
# plt.xlabel(r'$x_1$', fontsize = 15)
# plt.ylabel(r'$x_2$', fontsize = 15)
# plt.legend(loc = 1, fontsize = 12)
# plt.axis('equal')
# plt.xlim([0, 8])
# plt.ylim([-4, 3])
# plt.show()



### compact model, e.T = [u , v], svm model 2 ###

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N,1]), -np.ones([M,1])])
m = N + M
w = cvx.Variable([3,1])
d = cvx.Variable([m,1])

g = 1 # 클수록 mis 허용 x , 작을 수록 robust

obj = cvx.Minimize(cvx.norm(w,2) + g*(np.ones([1,m])*d))
const = [cvx.multiply(y, X*w) >= 1-d, d >= 0] #inner product(내적)가 아니므로
prob = cvx.Problem(obj, const).solve()
w = w.value

xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize = (10, 8))
plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.1, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.1)
plt.plot(xp, ypt+1, '--k', alpha = 0.1)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'SVM')
plt.plot(xp, yp-1/w[2,0], '--g')
plt.plot(xp, yp+1/w[2,0], '--g')
plt.title('Support Vector Machine (SVM)', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()
