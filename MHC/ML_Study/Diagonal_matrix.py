# Diagonal Matrix (대칭 행렬, 단위 행렬)

# [d1 0 0 0 0]
# [0 d2 0 0 0]
# [0 0 d3 0 0]
# [0 0 0 d4 0]
# [0 0 0 0 d5]

# identity matrix I = diag(1)
import numpy as np
d = np.array([1,2,3])
D = np.diag(d)

print(D)
