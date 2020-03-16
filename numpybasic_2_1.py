import numpy as np

data = [6, 7, 8, 0, 1]
arr1 = np.array(data)
print('data:', data)
print('arr1:', arr1)


data2 = [[1,2,3,4],[5,6,7,8]] ## 2x4 행렬
arr2 = np.array(data2)
print(data2)
print('arr2:\n',arr2)

print('ndim:', arr2.ndim) #몇차원인지 확인하는것
print('shape:', arr2.shape) #행렬의 shape 확인
print('dtype:', arr2.dtype) #data type


arr3 = np.array([1,2,3,4,5], dtype = np.float32) #float형태로 지정
print('arr3 dtype:', arr3.dtype)

arr4 = np.arange(10) #array에서 range사용
print('arr4:',arr4)

arr5 = np.zeros((3,4)) #0으로 채우는
print('arr5:\n',arr5)
