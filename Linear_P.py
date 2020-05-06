import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

f = np.array([[1], [1]])
A = np.array([[2, 1], [1, 2]])
b = np.array([[29], [25]])
lb = np.array([[2], [5]])

x = cv.Variable(2,1)

objective = cv.Minimize(-f.T*x)   # -f.T * x 실제론 이표현, 만약 f=np.array([[1],[1]]) 이와 같이 선언했을때 -f.T*x 가된다.
constraints = [A*x <= b, lb <= x]

prob = cv.Problem(objective, constraints)
result = prob.solve()

print('x=',x.value,'\n')
print('result=',result,'\n')
