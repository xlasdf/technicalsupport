# 3-3 수학 메서드와 통계 메서드

# 배열 전체 혹은 배열에서 한 축에 따르는 자료에 대한 통계를 계산하기 위해
# 수학 함수는 배열메서드로 사용할 수 있다.

# numpy 함수
# sum(),mean() : 배열 전체합, 평균
# cumsum(), cumprod() : 누적합, 누적곱
# std(),var() : 표준편차, 분산
# min(), max() : 최소, 최대
# argmin(), argmax(): 최소원소의 색인값, 최대원소의 색인값

import numpy as np

arr = np.random.randn(5,4)
print('\n',arr)
print('mean1:', arr.mean())
print('mean2:',np.mean(arr))   # 위와 동일한 표현
print('sum:', arr.sum())

print('\n', arr.mean(axis=1))   # axis 1은 각각 행벡터, axis 0는 각각 열벡터
print('\n', arr.sum(axis=0))

arr = np.array([0,1,2,3,4,5,6,7])
print('\n', arr.cumsum())  #누적된형태의 합.

arr = np.array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])

print('\n', arr.cumprod(axis=1))   #누적곱 # axis 1은 각각 행벡터, axis 0는 각각 열벡터
print('\n', arr.cumsum(axis=0))


# 불리언 배열을 위한 메서드
# numpy는 불리언 값을 이용한 배열의 값을 선택하고 처리할 수 있는 함수를 제공하고있음.
#
# arr = np.random.randn(100)
# (arr>0).sum()
#
# any, all 메서드는 불리언 배열에 사용할 때 유용하다.
# any 메서드는 하나 이상의 True 값이 있는지 검사하고, all메서드는 모든원소가 True 인지 검사한다.


arr = np.random.randn(100)
print(arr)
print('\n',(arr>0).sum(),'\n')

bools = np.array([[False, False, True, False],
                 [False, False, False, False]])

print(bools.any(axis=1),'\n')
print(bools.all(axis=0),'\n')

# 응용
data = np.random.randn(10, 4)
data = data * 2
print('\ndata:\n',data)
print('\ndata[(data>3).any(axis=1)]\n', data[(data>3).any(axis=1)])

#행의 값들중에서 3초과 되는 값이 하나라도 존재한다면 그 행을 가져와라
# any 함수를 이용해서 행, 열단위로 가져오게됨, 자주사용
