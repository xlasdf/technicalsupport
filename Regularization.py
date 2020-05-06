#Regularization (Shinkage Methods), lambda

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

x = np.linspace(-4.5, 4.5, 10).reshape(-1,1)
y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180, -0.8361, -0.6591, -2.4701, -2.8122, -6.2512]).reshape(-1,1)
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1,1)
d = 10
u = np.linspace(-4.5, 4.5, d)
sigma = 1

sel = int(input('Select Case (1:Overfitted, 2:ridge Regression, 3:lasso Regression, 4:selective Lasso theta'))
if(sel == 1):

    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])

    rbfbasis = np.asmatrix(rbfbasis)
    A = np.asmatrix(A)

    theta = cvx.Variable(10, 1)
    obj = cvx.Minimize(cvx.sum_squares(A*theta-y))
    prob = cvx.Problem(obj).solve()

    yp = rbfbasis*theta.value

    plt.figure(figsize = (10, 8))
    plt.plot(x, y, 'o', label = 'Data')
    plt.plot(xp, yp, label = 'Overfitted')
    plt.title('(Overfitted) Regression', fontsize = 15)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    plt.axis([-5, 5, -12, 6])
    plt.legend(fontsize = 15)
    plt.grid(alpha = 0.3)
    plt.show()

# ridge Regression
elif(sel == 2):

    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])

    rbfbasis = np.asmatrix(rbfbasis)
    A = np.asmatrix(A)
    lamb = 0.1  # 클수록 rounding

    theta = cvx.Variable(10, 1)

    obj = cvx.Minimize(cvx.sum_squares(A*theta-y) + lamb*cvx.norm(theta,2))
    prob = cvx.Problem(obj).solve()

    yp = rbfbasis*theta.value

    plt.figure(figsize = (10, 8))
    plt.plot(x, y, 'o', label = 'Data')
    plt.plot(xp, yp, label = 'Ridge')
    plt.title('Ridge Regression (L2)', fontsize = 15)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    plt.axis([-5, 5, -12, 6])
    plt.legend(fontsize = 15)
    plt.grid(alpha = 0.3)

    # theta plot
    plt.figure(2, figsize=(10,8))
    plt.title(r'Ridge: magnitude of $\theta$', fontsize=15)
    plt.xlabel(r'$\theta$', fontsize = 15)
    plt.ylabel('magnitude', fontsize = 15)
    plt.stem(np.linspace(1,10,10).reshape(-1,1), theta.value)
    plt.xlim([0.5, 10.5])
    plt.ylim([-5, 5])
    plt.grid(alpha=0.3)

    lamb_v = np.arange(0,3,0.01)

    theta_record = []
    for k in lamb_v:
        theta_v = cvx.Variable(10,1)
        obj_v = cvx.Minimize(cvx.sum_squares(A*theta_v-y) + k*cvx.norm(theta_v,2))
        prob_v = cvx.Problem(obj_v).solve()
        theta_record.append(np.ravel(theta_v.value))

    plt.figure(3, figsize=(10,8))
    plt.plot(lamb_v, theta_record, linewidth=1)
    plt.title('Ridge coefficients as a funcfion of regularization', fontsize=15)
    plt.xlabel('$\lambda$', fontsize=15)
    plt.ylabel(r'weight $\theta$', fontsize=15)
    plt.show()

# Lasso Regression
elif(sel == 3):
    lamb = 2
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])

    rbfbasis = np.asmatrix(rbfbasis)
    A = np.asmatrix(A)

    theta = cvx.Variable(10, 1)

    obj = cvx.Minimize(cvx.sum_squares(A*theta-y) + lamb*cvx.norm(theta,1))
    prob = cvx.Problem(obj).solve()

    yp = rbfbasis*theta.value
    plt.figure(figsize = (10, 8))
    plt.plot(x, y, 'o', label = 'Data')
    plt.plot(xp, yp, label = 'LASSO')
    plt.title('LASSO Regularization(L1)', fontsize = 15)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    plt.axis([-5, 5, -12, 6])
    plt.legend(fontsize = 15)
    plt.grid(alpha = 0.3)

    # theta plot
    plt.figure(2, figsize=(10,8))
    plt.title(r'LASSO: magnitude of $\theta$', fontsize=15)
    plt.xlabel(r'$\theta$', fontsize = 15)
    plt.ylabel('magnitude', fontsize = 15)
    plt.stem(np.linspace(1,10,10).reshape(-1,1), theta.value)
    plt.xlim([0.5, 10.5])
    plt.ylim([-5, 5])
    plt.grid(alpha=0.3)

    #lambda 값 변화 하면서

    lamb_v = np.arange(0,3,0.01)

    theta_record = []
    for k in lamb_v:
        theta_v = cvx.Variable(10,1)
        obj_v = cvx.Minimize(cvx.sum_squares(A*theta_v-y) + k*cvx.norm(theta_v,1))
        prob_v = cvx.Problem(obj_v).solve()
        theta_record.append(np.ravel(theta_v.value))

    plt.figure(3, figsize=(10,8))
    plt.plot(lamb_v, theta_record, linewidth=1)
    plt.title('LASSO coefficients as a funcfion of regularization', fontsize=15)
    plt.xlabel('$\lambda$', fontsize=15)
    plt.ylabel(r'weight $\theta$', fontsize=15)
    plt.show()

# 3번에서 도출된 nonzero theta를 찾아 해당 x값으로만 Regression진행
elif(sel == 4):
    d = 4
    u = np.array([-3.5, -2.5, 2.5, 4.5])
    sigma = 1

    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])

    rbfbasis = np.asmatrix(rbfbasis)
    A = np.asmatrix(A)

    theta = cvx.Variable(4,1)
    obj = cvx.Minimize(cvx.norm(A*theta-y,2))
    prob = cvx.Problem(obj).solve()

    yp = rbfbasis*theta.value

    plt.figure(figsize = (10, 8))
    plt.plot(x, y, 'o', label = 'Data')
    plt.plot(xp, yp, label = 'Sparse')
    plt.title('Regression with Selected Features', fontsize = 15)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    plt.axis([-5, 5, -12, 6])
    plt.legend(fontsize = 15)
    plt.grid(alpha = 0.3)
    plt.show()
