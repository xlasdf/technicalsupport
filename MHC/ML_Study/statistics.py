import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# random number generation
m = 100

# uniform distribution U(0,1)
x1 = np.random.rand(m)
print(x1.shape)

# uniform distribution U(a,b)
a = 1
b = 5
x2 = a+(b-a)*np.random.rand(m)

# standard normal (Gaussian) distribution N(0,1^2)
# x3 = np.random.normal(0, 1, m)
x3 = np.random.randn(m,1)
# print(x3)
# normal distribution N(5,2^2)
x4 = 5 + 2*np.random.randn(m,1)

# random integers
x5 = np.random.randint(1, 6, size = (1,m))


# statistics
# numerically understand statisticcs

x = np.random.rand(m,1)

#xbar = 1/m*np.sum(x, axis = 0)
#np.mean(x, axis = 0)
xbar = 1/m*np.sum(x)
np.mean(x)

# 분산
varbar = (1/(m - 1))*np.sum((x - xbar)**2)
np.var(x)


# various sample size m , m이 클수록 sample mean이 real mean에 수렴

m = np.arange(10, 2000, 20)
means = []

for i in m:
    x = np.random.normal(10, 30, i)
    means.append(np.mean(x))

plt.figure(figsize = (10,6))
plt.plot(m, means, 'bo', markersize = 4)
plt.axhline(10, c = 'k', linestyle='dashed')
plt.xlabel('# of smaples (= sample size)', fontsize = 15)
plt.ylabel('sample mean', fontsize = 15)
plt.ylim([0, 20])
# plt.show()


# Seems approximately Gaussian distributed
# numerically demostrate that sample mean follows the Gaussin distribution
N = 1000
m = np.array([10, 40, 160])   # sample of size m

S1 = []   # sample mean (or sample average)
S2 = []
S3 = []

for i in range(N):
    S1.append(np.mean(np.random.rand(m[0], 1)))
    S2.append(np.mean(np.random.rand(m[1], 1)))
    S3.append(np.mean(np.random.rand(m[2], 1)))

plt.figure(figsize = (10, 6))
plt.subplot(1,3,1), plt.hist(S1, 21), plt.xlim([0, 1]), plt.title('m = '+ str(m[0])), plt.yticks([])
plt.subplot(1,3,2), plt.hist(S2, 21), plt.xlim([0, 1]), plt.title('m = '+ str(m[1])), plt.yticks([])
plt.subplot(1,3,3), plt.hist(S3, 21), plt.xlim([0, 1]), plt.title('m = '+ str(m[2])), plt.yticks([])


# correlation coefficient

m = 300
x = np.random.rand(m)
y = np.random.rand(m)
#
# 순서대로 배열
xo = np.sort(x)
yo = np.sort(y)
yor = -np.sort(-y)

plt.figure(figsize = (8, 8))
plt.plot(x, y, 'ko', label = 'random')
plt.plot(xo, yo, 'ro', label = 'sorted')
plt.plot(xo, yor, 'bo', label = 'reversely ordered')

plt.xticks([])
plt.yticks([])
plt.xlabel('x', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.axis('equal')
plt.legend(fontsize = 12)

print(np.corrcoef(x,y), '\n')
print(np.corrcoef(xo,yo), '\n')
print(np.corrcoef(xo,yor))



# correlation coefficient
# 2배 했을때
m = 300
x = 2*np.random.randn(m)
y = np.random.randn(m)

xo = np.sort(x)
yo = np.sort(y)
yor = -np.sort(-y)

plt.figure(figsize = (8, 8))
plt.plot(x, y, 'ko', label = 'random')
plt.plot(xo, yo, 'ro', label = 'sorted')
plt.plot(xo, yor, 'bo', label = 'reversely ordered')

plt.xticks([])
plt.yticks([])
plt.xlabel('x', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.axis('equal')
plt.legend(fontsize = 12)

print(np.corrcoef(x,y), '\n')
print(np.corrcoef(xo,yo), '\n')
print(np.corrcoef(xo,yor))


d = {'col. 1': x, 'col. 2': xo, 'col. 3': yo, 'col. 4': yor}
df = pd.DataFrame(data = d)

sns.pairplot(df)
plt.show()
