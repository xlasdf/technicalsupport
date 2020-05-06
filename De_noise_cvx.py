import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

n = 200
t = np.arange(n).reshape(-1,1)
x = 0.5 * np.sin((2*np.pi/n)*t) * (np.sin(0.01*t))

x_cor = x + 0.05*np.random.randn(n,1)

# plt.figure(figsize = (10,8))
# plt.subplot(2,1,1)
# plt.plot(t,x,'-',linewidth=2)
# plt.xticks([])
# plt.title('original signal', fontsize=15)
# plt.ylabel('$x_{original}$', fontsize=15)
#
# plt.subplot(2,1,2)
# plt.plot(t, x_cor, '-', linewidth = 1)
# plt.axis([0, n, -0.6, 0.6])
# plt.title('corrupted signal', fontsize=15)
# plt.xlabel('n', fontsize = 15)
# plt.ylabel('$x_{corrupted}$', fontsize=15)

D = np.zeros([n-1,n])
D[:,0:n-1] = D[:,0:n-1] - np.eye(n-1)
D[:,1:n] = D[:,1:n] + np.eye(n-1)



#2norm

gammas = [0,2,7]

for i in range(len(gammas)):
    x_reconst = cvx.Variable(n,1)
    obj = cvx.Minimize(cvx.norm(x_reconst - x_cor,2)+ gammas[i]*cvx.norm(D*x_reconst,2))
    prob = cvx.Problem(obj).solve()
    plt.subplot(3,1,i+1)
    plt.plot(t, x_cor, '-', linewidth = 1, alpha = 0.3);
    plt.plot(t, x_reconst.value, 'r', linewidth = 2)
    plt.ylabel('$\gamma = {}$'.format(gammas[i]), fontsize = 15)
plt.show()

#sum_squares2
# mu = [0, 10, 100]
# for i in range(len(mu)):
#     x_reconst = cvx.Variable(n,1)
#     # obj = cvx.Minimize(cvx.sum_squares(x_reconst - x_cor) + mu*cvx.sum_squares(x_reconst[1:n]-x_reconst[0:n-1]))
#     obj = cvx.Minimize(cvx.sum_squares(x_reconst - x_cor) + mu[i]*cvx.sum_squares(D*x_reconst))
#     prob = cvx.Problem(obj).solve()
#
#     plt.subplot(3,1,i+1)
#     plt.plot(t, x_cor, '-', linewidth = 1, alpha = 0.3);
#     plt.plot(t, x_reconst.value, 'r', linewidth = 2)
#     plt.ylabel('$\mu = {}$'.format(mu[i]), fontsize = 15)
# plt.show()


#sum_squares1
# mu = 100
#
# x_reconst = cvx.Variable(n,1)
# # obj = cvx.Minimize(cvx.sum_squares(x_reconst - x_cor) + mu*cvx.sum_squares(x_reconst[1:n]-x_reconst[0:n-1]))
# obj = cvx.Minimize(cvx.sum_squares(x_reconst - x_cor) + mu*cvx.sum_squares(D*x_reconst))
# prob = cvx.Problem(obj).solve()
#
# plt.figure(figsize = (10, 4))
# plt.plot(t, x_cor, '-', linewidth = 1, alpha = 0.3, label = 'corrupted');
# plt.plot(t, x_reconst.value, 'r', linewidth = 2, label = 'reconstructed')
# plt.title('$\mu = {}$'.format(mu), fontsize = 15)
# plt.legend(fontsize = 12)
# plt.show()
