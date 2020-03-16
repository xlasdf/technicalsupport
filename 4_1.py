#선형대수

#numpy 라이브러리는 행렬의 곱셈, 분할, 행렬식, 정사각행렬 같은 선형대수 함수를 제공한다.
#MATLAB 같은 다른 언어와 달리 2개의 2차원배열을 * 연산자로 곱하는건 행렬곱셈이 아니라 대응하는 각각의 원소의 곱을 계산하는것.
#행렬곱셈은 배열 메서드이자 numpy 네임스페이스 안에 있는 함수인 dot 함수를 사용해서 계산한다.

import numpy as np

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[6,23],[1,-7],[8,9]])
print(x)
print('\n',y)
print('\n',x.dot(y))  #행렬곱

#선형 대수 함수
# numpy.diag   정사각행렬의 대각/비대각 원소를 1차원 배열로 반환하거나, 1차원배열을 대각선 원소로 하고 나머지는 0으로 채운 단위 행렬을 반환한다.
# numpy.dot    행렬곱셈
# numpy.trace  행렬의 대각선 원소의 합을 계산
# numpy.linalg.det  행렬식을 계산
# numpy.linalg.eig  정사각행렬의 고유값과 고유 벡터를 계산
# numpy.linalg.inv  정사각행렬의 역행렬 계산
# numpy.linalg.solve   A가 정사각행렬일때 Ax =b를 만족하는 x를 구함
# numpy.linalg.lstsq   y = xb를 만족하는 최소제곱해를 구함.

print('\n',np.ones(3))  # ones(3) 1X3행렬
print('\n',np.dot(x, np.ones(3)))

from numpy.linalg import inv, qr    #numpy.linalg 의 inv, qr 불러오기
x2 = np.random.randn(5,5)
mat = x2.T.dot(x2)
print('\n',inv(mat))   #np.linalg.inv와 동일

print('\n',mat)

q,r = qr(mat)    #q,r값 구하는 function,np.linalg.qr과 동일
print('q:\n',q)  # 행렬 QR 분해
print('r:\n',r)

data = np.identity(6) #6x6  I 단위 matrix
print('\n',data)
