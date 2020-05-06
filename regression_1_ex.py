import numpy as np
import matplotlib.pyplot as plt
# data points in column vector [input, output]
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/05_Regression_1-1.html
# https://philosopher-chan.tistory.com/461
x = np.array([0.2, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1,1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1,1)
# 3points x,y pair

m = y.shape[0] # 13개 행
A = np.hstack([np.ones([m,1]), x]) # horiziontal stack -> hstack, vertical stack -> vstack
# A = np.hstack([x**0,x])
A = np.asmatrix(A)
#theta = np.linalg.inv(A.T*A)*A.T*y
#theta = (A.T*A).I*A.T*y
theta = np.linalg.inv(A.T*A)*A.T*y

print('theta:\n',theta)


# to plot

plt.figure(figsize=(10,8))
plt.title('$L_2$ Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x,y,'ko', label="data")
#to plot a straight line(fitted line)

xp = np.arange(0, 5 ,0.01).reshape(-1,1)
yp = theta[0,0]*xp + theta[1,0]

plt.plot(xp, yp, 'r', linewidth=2, label="$L_2$")
plt.legend(fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.xlim([0,5])
plt.show()
