import numpy as np

# arr = np.array([[1,2,3],[4,5,6]])
# print('arr:\n',arr)
# print('arr*arr:\n',arr*arr) # 각각원소 제곱
# print('arr-arr:\n',arr-arr) # 자기 빼기 자기
#
# print('1/arr:\n',1/arr)
# print(('arr ** 0.5:\n', arr ** 0.5)) # root
#

arr2 = np.arange(10) #0,1,2,3,4,5,...9

print('\n',arr2[5])
print('\n', arr2[5:8])
arr2[5:8] = 12

print('\n', arr2)

arr3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('\n',arr3[:3])  #arr[2,0] ,기준으로 행과 열 나눔, 이는 즉 0,1,2행 전부 출력해라.
print('\n',arr3[:1,:3])
print('\n',arr3[:2,1:]) # arr3의 0~1행까지와 0열 제외 1열부터 끝까지 출력해라.


##불리언 배열

# 축의 길이와 동일한 길이를 가지는 불리언 배열
np.random.seed(12345)  #seed를 사용하여 일정한

names = np.array(['Bob', 'Joe', 'Will', 'Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
print('\n',names, "\n")
print('\n',data)
print('\n',names=='Bob') # names에 Bob 이 있을 때 True
print('\n',data[names=='Bob'])  #True 값을 찾아서 data에서 True값에 매칭되는 행을 출력
print('\n',data[names=='Bob',2:])  #True 값을 찾아서 data에서 True값에 매칭되는 행과 2번째 열부터 출력
print('\n',data[names=='Bob',3])  #True 값을 찾아서 data에서 True값에 매칭되는 행과 3열을 출력
