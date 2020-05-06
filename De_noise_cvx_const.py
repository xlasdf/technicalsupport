import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

n = 200
t = np.arange(n).reshape(-1,1)
x = 0.5 * np.sin((2*np.pi/n)*t) * (np.sin(0.01*t))

x_cor = x + 0.05*np.random.randn(n,1)

D = np.zeros([n-1,n])
D[:,0:n-1] = D[:,0:n-1] - np.eye(n-1)
D[:,1:n] = D[:,1:n] + np.eye(n-1)

beta = [0.1, 0.6, 0.8]
for i in range(len(beta)):
    x_reconst = cvx.Variable(n,1)
    obj = cvx.Minimize(cvx.norm(D*x_reconst,2))
    const = [cvx.norm(x_reconst - x_cor,2)<= beta[i]]
    prob = cvx.Problem(obj, const).solve()

    plt.subplot(len(beta),1,i+1)
    plt.plot(t,x_cor,'-',linewidth=1, alpha=0.3)
    plt.plot(t,x_reconst.value, 'r',linewidth=2)
    plt.ylabel(r'$\beta = {}$'.format(beta[i]), fontsize =15)
plt.show()



# beta = 0.8
# x_reconst = cvx.Variable(n,1)
# obj = cvx.Minimize(cvx.norm(D*x_reconst,2))
# const = [cvx.norm(x_reconst - x_cor,2)<= beta]
# prob = cvx.Problem(obj, const).solve()
#
# plt.figure(figsize = (10,4))
# plt.plot(t,x_cor,'-',linewidth=1, alpha=0.3)
# plt.plot(t,x_reconst.value, 'r',linewidth=2)
# plt.ylabel(r'$\beta = {}$'.format(beta), fontsize =15)
# plt.show()
