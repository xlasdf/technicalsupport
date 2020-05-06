# https://twlab.tistory.com/46?category=668741 , Eigen value, vector 참조 글

# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/11_PCA.html
# 축을 찾는 것!


import numpy as np
import matplotlib.pyplot as plt

# data generation
m = 5000
mu = np.array([0, 0])
sigma = np.array([[3, 1.5],
                  [1.5, 1]])

X = np.random.multivariate_normal(mu, sigma, m) # 평균이 0 0 인 5000개  data
X = np.asmatrix(X)
print(X)
# X.shape = (5000*2)

S = 1/(m-1)*X.T*X #(2x2)
D, U = np.linalg.eig(S) # 고유값, 고유 벡터
print(D,'\n') #lambda, 즉 고유값
print(U, '\n') # 고유 벡터

idx = np.argsort(-D) # 고유값 큰 것 부터

D = D[idx] # 큰 것부터
U = U[:, idx]
print(D,'\n') #lambda, 즉 고유값
print(U, '\n') # 고유 벡터

h = U[1,0]/U[0,0] # u1 vetor
h2 = U[1,1]/U[0,1] # u2 vetor
xp = np.arange(-6,6,0.1)
yp = h*xp
yp2 = h2*xp

# projection method -> Z =U.T*X

z = X*U[:,0]



fig = plt.figure(figsize = (10, 8))
plt.plot(X[:,0], X[:,1], 'k.', alpha = 0.3)
plt.plot(xp, yp, 'r', linewidth=3)
# plt.plot(xp, yp2, 'b', linewidth=3)
plt.axis('equal')
plt.grid(alpha = 0.3)

plt.figure(figsize = (10, 8))
plt.hist(z, 51)

plt.show()


'''
from sklearn.decomposition import PCA

pca = PCA(n_components = 1)
pca.fit(X)

u = pca.transform(X)

plt.figure(figsize = (10, 8))
plt.hist(u, 51)
plt.show()
'''
#https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js
