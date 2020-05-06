import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/09_Clustering.html

# sklearn 을 활용해서 kmeans score 구하기, Elbow Point
# data generation
G0 = np.random.multivariate_normal([1,1], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)
G1 = np.random.multivariate_normal([3,5], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)
G2 = np.random.multivariate_normal([9,9], np.eye(2), 100) # multivariate_normal(평균, 모양, 개수)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)
print(X.shape)

cost = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0).fit(X)
    cost.append(abs(kmeans.score(X)))

plt.figure(figsize = (10,8))
plt.stem(range(1,11), cost)
plt.xticks(np.arange(11))
plt.xlim([0.5, 10.5])
plt.grid(alpha = 0.3)
plt.show()
