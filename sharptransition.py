import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

n = 200
t = np.arange(n).reshape(-1,1)

exact = np.vstack([np.ones([50,1]), -np.ones([50, 1]), np.ones([50,1]), -np.ones([50,1])])
x = exact + 0.5*np.sin((2*np.pi/n)*t)
x_cor = x + 0.1*np.random.randn(n,1)


# plt.figure(figsize=(10,8))
# plt.subplot(2,1,1)
# plt.plot(t,x)
# plt.ylim([-2.0,2.0])
# plt.ylabel('signal',fontsize=15)
# plt.subplot(2,1,2)
# plt.plot(t, x_cor, linewidth=1)
# plt.ylabel('corrupted signal', fontsize=15)
# plt.xlabel('x', fontsize=15)
# plt.show()

#2norm
plt.figure(figsize=(10,12))
beta = [0.5, 2, 4]
for i in range(len(beta)):
    x_reconst = cvx.Variable(n,1)
    obj = cvx.Minimize(cvx.norm(x_reconst[1:n]-x_reconst[0:n-1],2))
    const = [cvx.norm(x_reconst - x_cor, 2) <= beta[i]]
    prob = cvx.Problem(obj, const).solve()

    plt.subplot(len(beta), 1, i+1)
    plt.plot(t, x_cor, linewidth=1, alpha=0.3)
    plt.plot(t, x_reconst.value,'r', linewidth=2)
    plt.suptitle('2norm')
    plt.ylabel(r'$\beta = {}$'.format(beta[i]), fontsize=15)


#1norm
plt.figure(2, figsize=(10,12))
beta = [0.5, 2, 4]
for i in range(len(beta)):
    x_reconst = cvx.Variable(n,1)
    obj = cvx.Minimize(cvx.norm(x_reconst[1:n]-x_reconst[0:n-1],1))
    const = [cvx.norm(x_reconst - x_cor, 2) <= beta[i]]
    prob = cvx.Problem(obj, const).solve()

    plt.subplot(len(beta), 1, i+1)
    plt.plot(t, x_cor, linewidth=1, alpha=0.3)
    plt.plot(t, x_reconst.value,'r', linewidth=2)
    plt.suptitle('1norm')
    plt.ylabel(r'$\beta = {}$'.format(beta[i]), fontsize=15)
plt.show()
