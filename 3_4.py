# 정렬 및 집합 함수

# 정렬, numpy 에서 또한 sort 함수 사용가능, 내장함수에서도 가능

import numpy as np

arr = np.random.randn(3,3)
print('\n',arr)

arr.sort(axis=1)  #  axis 1은 각각 행벡터, axis 0는 각각 열벡터
print('\n', arr)

arr.sort(axis=0)

print('\n', arr)


#집합 함수
# 1차원 ndarray를 위한 집합 연산을 제공

# unique(x) 배열 x에서 중복된 원소를 제거한 후 정렬하여 반환한다.  #python set과 같은 자료구조, 다른점은 정렬을 한다는것!!
# intersect1d(x,y) 배열 x와 y에 공통적으로 존재하는 원소를 정렬하여 반환한다. #교집합
# union1d(x,y)  두배열의 합집합을 반환한다.
# in1d(x,y) x의 원소 중 y의 원소를 포함하는지를 나타내는 불리언 배열을 반환한다.
# setdiff1d(x,y) x와 y의 차집합을 반환한다.
# setxor1d(x,y) 한배열에는 포함되지만 두배열 모두에는 포함되지 않는 원소들의 집합인 대칭 차집합을 반환한다. #xor exclusive


intData = np.array([3,3,3,3,2,2,2,1,1,1,1,4,4])

print('\nunique:', np.unique(intData))

values = np.array([6,0,0,3,2,5,6])

print('\nunion:', np.union1d(values,[2,3,6])) #values 와 [2,3,6] 의 합집합

print('\nintersection:', np.intersect1d(values,[2,3,6])) #교집합
print('\ndifference:',np.setdiff1d(values,[2,3,6]))  # 차집합
