import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

x = np.linspace(-4.5, 4.5, 10).reshape(-1,1)
y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180, -0.8361, -0.6591, -2.4701, -2.8122, -6.2512]).reshape(-1,1)
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1,1)

sel = int(input('Select Case (1:linear, 2:Polynomial, 3:Nonlinear Regression with Polynomial(degree),4:overfit RBF Basis):, 5:RBF(basis num)'))

if (sel == 1):

    n = 10

    # A = np.hstack([x**0,x]) # linear Regression
    A = np.hstack([x**0, x, x**2]) # 2차
    A = np.asmatrix(A)
    theta = np.linalg.inv(A.T*A)*A.T*y


    yp = theta[0,0]+ theta[1,0]*xp + theta[2,0]*xp**2
    plt.figure(figsize = (10,8))
    plt.plot(x,y,'o', label= 'Data')
    plt.plot(xp, yp, linewidth=2, label='Linear')
    #plt.plot(xp[:,0], yp[:,0], line~~)
    plt.title('Linear Regression')
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.legend(fontsize=15, loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

#Polynomial overfit
elif(sel==2):
    A = np.hstack([x**i for i in range(10)])
    A = np.asmatrix(A)
    theta = np.linalg.inv(A.T*A)*A.T*y

    xp = np.arange(-4.5, 4.5, 0.01).reshape(-1,1)
    polybasis = np.hstack([xp**i for i in range(10)])
    polybasis = np.asmatrix(polybasis)

    yp = polybasis*theta

    plt.figure(figsize=(10,8))
    plt.plot(x,y, 'o', label='Data')
    plt.plot(xp, yp, linewidth=2, label='9th degree')
    plt.title('Overfitted', fontsize=15)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.legend(fontsize=15, loc='lower right')
    plt.grid(alpha = 0.3)
    plt.show()

#degree Case
elif(sel==3):
    d = [1, 3, 5, 9]
    RSS = [] #Residual Sum of Squares
    plt.figure(figsize=(12,10))
    plt.suptitle('Nonlinear Regression', fontsize=15)
    for k in range(4):
        A = np.hstack([x**i for i in range(d[k]+1)])
        polybasis = np.hstack([xp**i for i in range(d[k]+1)])

        A = np.asmatrix(A)
        polybasis = np.asmatrix(polybasis)

        theta = np.linalg.inv(A.T*A)*A.T*y
        yp = polybasis*theta

        RSS.append(np.linalg.norm(y - A*theta, 2)**2)

        plt.subplot(2, 2, k+1)
        plt.plot(x, y, 'o')
        plt.plot(xp, yp)
        plt.axis([-5, 5, -12, 6])
        plt.title('degree = {}'.format(d[k]))
        plt.grid(alpha=0.3)


    plt.figure(2, figsize=(10,8))
    plt.stem(d, RSS, label='RSS')
    plt.title('Residual Sum of Squares', fontsize=15)
    plt.xlabel('degree', fontsize=15)
    plt.ylabel('RSS', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.show()

elif(sel==4):
    d=10
    u = np.linspace(-4.5,4.5,d)
    sigma=5
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    rbfbasis = np.asmatrix(rbfbasis)
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.asmatrix(A)
    theta = np.linalg.inv(A.T*A)*A.T*y
    yp = rbfbasis*theta

    plt.figure(figsize=(10,8))
    plt.plot(x,y, 'o', label='Data')
    plt.plot(xp, yp, linewidth=2, label='9th degree')
    plt.title('(Overfitted) Regression with RBF basis', fontsize=15)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(alpha = 0.3)
    plt.show()

elif(sel==5):
    d = [2, 4, 6, 10]
    sigma=1
    plt.figure(figsize=(12,10))

    for k in range(4):
        u = np.linspace(-4.5,4.5,d[k])

        rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d[k])])
        rbfbasis = np.asmatrix(rbfbasis)
        A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d[k])])
        A = np.asmatrix(A)

        theta = np.linalg.inv(A.T*A)*A.T*y
        yp = rbfbasis*theta

        plt.subplot(2, 2, k+1)
        plt.plot(x, y, 'o')
        plt.plot(xp, yp)
        plt.axis([-5, 5, -12, 6])
        plt.title('num RBFs = {}'.format(d[k]), fontsize=10)
        plt.grid(alpha=0.3)

    plt.suptitle('Nonlinear Regression with RBF Functions', fontsize=15)
    plt.show()
