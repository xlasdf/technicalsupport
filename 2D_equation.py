import numpy as np

A = np.array([[4, -5],[-2,3]])
b = np.array([[-13],[9]])

x = np.linalg.inv(A).dot(b)   # linalg :선형 대수식을 사용하겠다. linalg.dot -> 선형대수중 내적연산을 사용하겠다.
print(x)
##행렬 연산이 직관적이지 않으므로!
A = np.asmatrix(A)
b = np.asmatrix(b)

x = A.I*b  # type을 변환함
print(x)

#inner product  X^T . Y = xi . yi(i = 1~n)

X = np.array([[1],[1]])
Y = np.array([[2],[3]])

IP = X.T.dot(Y)

print(IP)

## 역시나 asmatrix 사용

x = np.asmatrix(X)
y = np.asmatrix(v)

z = x.T*y

print(z)


A2 = np.array([[1,1],[3,2]])

A2T = np.linalg.inv(A2)

print(A2T)
