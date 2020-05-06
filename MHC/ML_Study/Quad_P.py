import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

f = np.array([[3],[4]])
H = np.array([[1,0],[0,0]])
A = np.array([[-1,-3],[2,5],[3,4]])
b = np.array([[-15],[100],[80]])
lb = np.array([[0],[0]])

x = cvx.Variable([2,1])

objective = cvx.Minimize(1/2*cvx.quad_form(x, H) + f.T*x)
constraints = [A*x <= b, lb<=x]

prob = cvx.Problem(objective, constraints)
result = prob.solve()


print('x=',x.value)
print('C=',result)
