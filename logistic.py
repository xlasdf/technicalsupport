import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/08_Logistic_Regression.html

# outlier 가 없는 이유 sigmoid 함수에 따라 값을 결정

# # sigmoid function
# z = np.linspace(-4,4,100)
# s = 1/(1+ np.exp(-z))
#
# plt.figure(figsize = (10,2))
# plt.plot(z,s)
# plt.xlim([-4, 4])
# plt.axis('equal')
# plt.grid(alpha=0.3)
# plt.show()


# data plot 하기
m = 100

w = np.array([[-6], [2], [1]])
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
y[C0] = 0 #compact model 에서는 -1

# print(y)

#logistic function이용해서 w 구하기
#log likelihood -> sum(Log(pi)) + sum(Log(1-pi)), i = 1~q, i = 1-q ~ m,
w = cvx.Variable([3,1])
obj = cvx.Maximize(y.T*X*w - cvx.sum(cvx.logistic(X*w)))
# obj = cvx.Minimize(y.T*X*w + cvx.sum(cvx.logistic(X*w))) #위와 동일표현
# y 를 Transpose 해서 X 와 연산하면 1제외 0 은 모두 0이므로 p개만 남게된다.
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
