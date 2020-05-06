import numpy as np

x = np.array([[4],[3]])

l2 = np.linalg.norm(x,2) #x2 norm 구하는 방법  l2 norm = sqr{(x1)^2 + (x2)^2 + (x3)^2 ... (xn)^2}

l1 = np.linalg.norm(x,1) #x1 norm 구하는 방법  l1 norm = |x1| + |x2| + |x3| ... |xn|


#orthogonality

x = np.matrix([[1],[2]])
y = np.matrix([[2],[-1]])
x.T*y
