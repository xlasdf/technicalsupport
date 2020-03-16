#배열 전치와 축이동

# 팬시 색인은 정수 배열을 사용한 색인을 설명하기위해 numpy에서 차용한 단어

import numpy as np
arr = np.empty((8,4))           #
print(arr)

for i in range(8):
    arr[i] = i
print('\n',arr)
print('\n',arr[[4,3,0,6]]) #4, 3,0,6 번째 행 출력
print('\n',arr[[-3,-5,-7]]) #-3,-5,-7번째 행 출력


arr1 = np.arange(8)
arr2 = arr1.reshape((4,2))
print('\n',arr1)
print('\n', arr2)

arr3 = arr1.reshape((4,2)).reshape((2,4))
print('\n',arr3)

# arr4 = np.zeros((4,2))
# print('\n',arr4)

arr4 = np.arange(15).reshape((3,5))
print('\n',arr4)
print('\n', arr4.T)
