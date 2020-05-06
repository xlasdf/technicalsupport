import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


# data plot 하기
m = 100

w = np.array([[-6], [2], [1]]) # -6*1 + 2*1 + 1*4 => X*w, -6 ~ 6 까지 만들기
X = np.hstack([np.ones([m,1]), 4*np.random.rand(m,1), 4*np.random.rand(m,1)])

w = np.asmatrix(w)
X = np.asmatrix(X)

y = 1/(1 + np.exp(-X*w)) > 0.5
# 0~1 이므로 중간값 0.5 보다 큰지 작은지

C1 = np.where(y==True)[0]
C0 = np.where(y==False)[0]
# y 가해당하는 위치를 C1과 C0 에 할당

y = np.empty([m,1]) # empty 함수는 mx1 만큼의 비어있는 배열 만들어준다.

y[C1] = 1
y[C0] = -1 #compact model 에서는 -1

y = np.asmatrix(y)

w = cvx.Variable([3,1])

obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(y,X*w))))
prob = cvx.Problem(obj).solve()

w = w.value

xp = np.linspace(0,4,100).reshape(-1,1)
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
plt.xlim([0,4])
plt.ylim([0,4])
plt.show()
