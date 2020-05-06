import numpy as np
import matplotlib.pyplot as plt

# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/09_Clustering.html

# data generation
G0 = np.random.multivariate_normal([1,1], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)
G1 = np.random.multivariate_normal([3,5], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)
G2 = np.random.multivariate_normal([9,9], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)
print(X.shape)



# The number of clusters and data

k = 3
m = X.shape[0]

# randomly initialize mean points, 즉 centroid 설정
mu = X[np.random.randint(0,m,k),:] # X 중 3point 저장
pre_mu = mu.copy() # update 전 mu(centroid)를 저장
print(mu)

y = np.empty([m,1]) # assign 공간 형성

# Run K-means , iteration(반복) 진행

for n_iter in range(500):
    for i in range(m):
        d0 = np.linalg.norm(X[i,:] - mu[0,:],2)     # 각 centroid와 X 간 거리 계산
        d1 = np.linalg.norm(X[i,:] - mu[1,:],2)
        d2 = np.linalg.norm(X[i,:] - mu[2,:],2)
        y[i] = np.argmin([d0, d1, d2]) # 각 거리 최소값을 토대로 assign 진행, argmin은 최소인 위치를 할당한다. 즉 d0가 최소이면 y = 0, d1 최소이면 y=1,
        # mean값(centroid) 갱신하기
    err=0
    for i in range(k):
        mu[i,:] = np.mean(X[np.where(y==i)[0]], axis = 0) #y가 0~2으로 indexing 된 값들을 토대로 평균 구해서 centroid 갱신, 열방향 평균
        err += np.linalg.norm(pre_mu[i,:] - mu[i,:],2) # 이전 에러값과 갱신에러 차이계산하기
    pre_mu = mu.copy()

    if err < 1e-10:
        print("iteration:", n_iter)
        break
# assign 된 data로 쪼개기
X0 = X[np.where(y==0)[0]]
X1 = X[np.where(y==1)[0]]
X2 = X[np.where(y==2)[0]]

plt.figure(figsize = (10, 8))
plt.plot(X0[:,0], X0[:,1], 'b.', label = 'C0')
plt.plot(X1[:,0], X1[:,1], 'g.', label = 'C1')
plt.plot(X2[:,0], X2[:,1], 'r.', label = 'C2')
plt.axis('equal')
plt.legend(fontsize = 12)
plt.show()
