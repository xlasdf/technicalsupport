# 유니버셜 함수
#ufunc라고 불리는 유니버셜 함수는 ndarray안에 있는 데이터 원소별로 연산을 수행하는 함수

# 사칙연산 add(),multiply(),subtract(),div()
# 삼각함수 sin(),cos(),hypot()
# 비트단위 bitwise_and(), left_shift()
# 관계형, 논리 less(),logical_not(), equal()
# maximum()과 minimum(), modf()
# 부동소수점에 적용할 수 있는 함수 : isinf(), infinite(), floor(), isnan()

import numpy as np

arr = np.arange(10)
a = np.sqrt(arr)
print('\n',a)
b = np.exp(arr)
print('\n',b)

np.random.seed(12345)
x = np.random.randn(8)
y = np.random.randn(8)
print('x:',x)
print('y:',y)
print('maximum:',np.maximum(x,y)) # x,y 값비교해서 큰값 출력, 2항 ufunc 함수
