import numpy as np
import matplotlib.pyplot as plt
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/06_Perceptron.html

m = 100 # 100개의 data point

x1 = 8*np.random.rand(m,1)
x2 = 7*np.random.rand(m,1) - 4

g = 0.8*x1 + x2 -3

C0 = np.where(g >= 1)
C1 = np.where(g < -1)  # Class 분류

C0 = np.where(g >= 1)[0]
C1 = np.where(g < -1)[0]
#C0.shape, C1.shape
# print(C0)
# print(C1)

# 100개의 data point plot
# plt.figure(figsize=(10,8))
# plt.plot(x1[C0],x2[C0],'ro',alpha=0.4, label='c0')
# plt.plot(x1[C1],x2[C1],'bo',alpha=0.4, label='c1')
# plt.title('Linearly Separable Classes', fontsize=15)
# plt.legend(loc=1, fontsize=15)
# plt.xlabel(r'$x_1$', fontsize=15)
# plt.ylabel(r'$x_2$', fontsize=15)
# plt.show()

N = C0.shape[0]
M = C1.shape[0]

X0 = np.hstack([np.ones([N,1]), x1[C0], x2[C0]])
X1 = np.hstack([np.ones([M,1]), x1[C1], x2[C1]])

X = np.vstack([X0,X1])
y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

X = np.asmatrix(X)
y = np.asmatrix(y)

w = np.ones([3,1]) # ones or zeros
w = np.asmatrix(w)

n_iter = N+M
# print(X.shape)
# print(X.T.shape)

fin = 0
print((X[1,:]*w))
print((X[1,:]*w)[0,0],'\n') #위 연산에서 값만 추출하는것

## w = w + y*x, for문 100번 반복문
# for _ in range(100):
#     for i in range(n_iter):
#         if y[i,0] != np.sign(X[i,:]*w)[0,0]:
#             w = w + y[i,0]*X[i,:].T
# print(w)

## while 문으로
while(fin!=100):
    fin +=1
    for i in range(n_iter):
        if y[i,0] != np.sign(X[i,:]*w)[0,0]:
            w = w + y[i,0]*X[i,:].T

print(w)
#계산된 w로 직선 만들기
x1p = np.linspace(0,8,100).reshape(-1,1)
x2p = -w[1,0]/w[2,0]*x1p -w[0,0]/w[2,0] # w0 + w1x1 + w2x2 = 0 -> x2 = -w1/w2*x1 - w0/w2

plt.figure(figsize=(10,8))
plt.plot(x1[C0],x2[C0],'ro',alpha=0.4, label='c0')
plt.plot(x1[C1],x2[C1],'bo',alpha=0.4, label='c1')
plt.plot(x1p,x2p,c='k',linewidth=3, label='perceptron')
plt.xlim([0,8])
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.show()



'''
Skit Learn으로 perceptron 구현하기!

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])
X = np.vstack([X1, X0])

y = np.vstack([np.ones([C1.shape[0],1]), -np.ones([C0.shape[0],1])])

from sklearn import linear_model

clf = linear_model.Perceptron(tol=1e-3)
clf.fit(X, np.ravel(y))

clf.predict([[3, -2]])

clf.predict([[6, 2]])
clf.coef_
clf.intercept_

w0 = clf.intercept_[0]
w1 = clf.coef_[0,0]
w2 = clf.coef_[0,1]
x1p = np.linspace(0,8,100).reshape(-1,1)
x2p = - w1/w2*x1p - w0/w2

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, c = 'k', linewidth = 4, label = 'perceptron')
plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.show()

'''
