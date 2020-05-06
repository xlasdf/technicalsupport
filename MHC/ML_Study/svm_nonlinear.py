import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

np.set_printoptions(threshold=np.inf, linewidth=np.inf)# numpy ... 생략 제거

X1 = np.array([[-1.1,0],[-0.3,0.1],[-0.9,1],[0.8,0.4],[0.4,0.9],[0.3,-0.6],
               [-0.5,0.3],[-0.8,0.6],[-0.5,-0.5]])

X0 = np.array([[-1,-1.3], [-1.6,2.2],[0.9,-0.7],[1.6,0.5],[1.8,-1.1],[1.6,1.6],
               [-1.6,-1.7],[-1.4,1.8],[1.6,-0.9],[0,-1.6],[0.3,1.7],[-1.6,0],[-2.1,0.2]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N,1]), -np.ones([M,1])])
X = np.asmatrix(X)
y = np.asmatrix(y)

m = N + M
Z = np.hstack([np.ones([m,1]), np.square(X[:,0]), np.sqrt(2)*np.multiply(X[:,0],X[:,1]), np.square(X[:,1])]) # z(hstack) = [1, X1^2, root(2)*X1*X2, X2^2]

g = 10 # 클수록 mis 허용 x , 작을 수록 robust

w = cvx.Variable([4,1])
d = cvx.Variable([m,1])

obj = cvx.Minimize(cvx.norm(w,2) + g*(np.ones([1,m])*d))
const = [cvx.multiply(y, Z*w) >= 1-d, d >= 0] #inner product(내적)가 아니므로
prob = cvx.Problem(obj, const).solve()
w = w.value

print(w)

[X1gr, X2gr] = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1)) # 그래프 x, y 영역 모든 그리드 생성 3600 x 3600
Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)]) # 위 해당하는 모든 점
Xp = np.asmatrix(Xp)
m = Xp.shape[0]

Zp = np.hstack([np.ones([m,1]), np.square(Xp[:,0]), np.sqrt(2)*np.multiply(Xp[:,0],Xp[:,1]), np.square(Xp[:,1])])
#-3~3, -3~3 모든 평면을 그린다.

q = Zp*w

B = []

for i in range(m):
    if q[i,0] > 0:   # C1 에 해당하는 것,C1(g1 >= 0) outlier 는 
        B.append(Xp[i,:])
B = np.vstack(B)

plt.figure(figsize=(10,8))
plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
plt.plot(B[:,0], B[:,1], 'gs', markersize = 10, alpha=0.1, label = 'SVM')
plt.title('SVM with Kernel', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.show()
