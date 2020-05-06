#https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/07_SVM.html
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

# w0 + wTx > 0, w0 + wTx < 0 -> band를 넓혀서, w0 + wTx > 1, w0 + wTx < -1
# plot 하기
# plt.figure(figsize = (10, 8))
# plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
# plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
# plt.plot(xp, ypt, 'k', linewidth = 3, label = 'True')
# plt.plot(xp, ypt-1, '--k') # -1 만큼
# plt.plot(xp, ypt+1, '--k') # 1 만큼
#
# plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
# plt.xlabel(r'$x_1$', fontsize = 15)
# plt.ylabel(r'$x_2$', fontsize = 15)
# plt.legend(loc = 1, fontsize = 12)
# plt.axis('equal')
# plt.xlim([0, 8])
# plt.ylim([-4, 3])
# plt.show()

# 최적화
# #Form 1 :  w0 + X1w >= 1, w0 + X0w <= -1

# X1 = np.hstack([x1[C1], x2[C1]])
# X0 = np.hstack([x1[C0], x2[C0]])
#
# X1 = np.asmatrix(X1)
# X0 = np.asmatrix(X0)
#
# N = X1.shape[0]
# M = X0.shape[0]
#
# w0 = cvx.Variable([1,1])
# w = cvx.Variable([2,1])
#
# obj = cvx.Minimize(1)
# const = [w0 + X1*w >= 1, w0 + X0*w <= -1]
# prob = cvx.Problem(obj, const).solve()
#
# w0 = w0.value
# w = w.value
#
# print(w0,'\n',w)
# yp = - w[0,0]/w[1,0]*xp - w0/w[1,0] # w0 + w1x1 + w2x2 = 0 -> x2 = -w1/w2*x1 - w0/w2
#
# plt.figure(figsize = (10, 8))
# plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
# plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
# plt.plot(xp, ypt, 'k', linewidth = 3, label = 'True')
# plt.plot(xp, ypt-1, '--k', alpha=0.3) # -1 만큼
# plt.plot(xp, ypt+1, '--k', alpha=0.3) # 1 만큼
# plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 1')
# plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
# plt.xlabel(r'$x_1$', fontsize = 15)
# plt.ylabel(r'$x_2$', fontsize = 15)
# plt.legend(loc = 1, fontsize = 12)
# plt.axis('equal')
# plt.xlim([0, 8])
# plt.ylim([-4, 3])
# plt.show()

## Form 2 :  X1w >= 1,  X0w <= -1

N = C1.shape[0]
M = C0.shape[0]

X1 = np.hstack([np.ones([N,1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([M,1]), x1[C0], x2[C0]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

w = cvx.Variable([3,1])

obj = cvx.Minimize(1)
const = [X1*w >= 1, X0*w <= -1]
prob = cvx.Problem(obj, const).solve()

w = w.value

xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize = (10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.3, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.3)
plt.plot(xp, ypt+1, '--k', alpha = 0.3)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 1')
plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()
