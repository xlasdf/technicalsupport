import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

n = 4000
t = np.arange(n).reshape(-1,1)
x = 0.5 * np.sin((2*np.pi/n)*t) * (np.sin(0.01*t))

x_cor = x + 0.05*np.random.randn(n,1)

plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
plt.plot(t,x,'-',linewidth=2)
plt.xticks([])
plt.title('original signal', fontsize=15)
plt.ylabel('$x_{original}$', fontsize=15)

plt.subplot(2,1,2)
plt.plot(t, x_cor, '-', linewidth = 1)
plt.axis([0, n, -0.6, 0.6])
plt.title('corrupted signal', fontsize=15)
plt.xlabel('n', fontsize = 15)
plt.ylabel('$x_{corrupted}$', fontsize=15)

D = np.zeros([n-1,n])
D[:,0:n-1] = D[:,0:n-1] - np.eye(n-1)
D[:,1:n] = D[:,1:n] + np.eye(n-1)

mu = [0, 10, 100]

for i in range(len(mu)):
    A = np.vstack([np.eye(n), np.sqrt(mu[i])*D])
    b = np.vstack([x_cor, np.zeros([n-1, 1])])

    A = np.asmatrix(A)
    b = np.asmatrix(b)

    x_reconst = np.linalg.inv(A.T*A)*A.T*b

    plt.figure(2)
    plt.subplot(3,1,i+1)
    plt.plot(t, x_cor, '-', linewidth=1, alpha=0.3)
    plt.plot(t, x_reconst, 'r', linewidth=2)
    plt.ylabel('$\mu = {}$'.format(mu[i]), fontsize=15)

plt.show()



# mu = 100
#
# A = np.vstack([np.eye(n), np.sqrt(mu)*D])
# b = np.vstack([x_cor, np.zeros([n-1, 1])])
#
# A = np.asmatrix(A)
# b = np.asmatrix(b)
#
# x_reconst = np.linalg.inv(A.T*A)*A.T*b
#
# plt.figure(2, figsize=(10,4))
# plt.plot(t, x_cor, '-', linewidth=1, alpha=0.3, label = 'corrupted')
# plt.plot(t, x_reconst, 'r', linewidth=2, label='reconstructed')
# plt.title('$\mu = {}$'.format(mu), fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
