#조건부 함수

# where(c,a,b) 함수 : x if 조건 else y 같은 삼항식의 벡터화 된 버전.
# any(), all() 함수 : 각각 일부 혹은 모든 배열의 엘리먼트가 True라면 True를 반환

import numpy as np

xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])

result = [(x if c else y) for x,y,c in zip (xarr,yarr,cond)]    #x if c else y : c를 기준으로 true면 x 아니면 y
print('result:', result)
result2 = np.where(cond,xarr,yarr) # 위 result 결과와 같은 표현, True면 x, False면 y, -> where 조건함수
print('\nresult2:',result2)

np.random.seed(12345)
arr = np.random.randn(4,4)
print('arr:\n',arr)
print('\n',np.where(arr>0,2,-2))   # condition으로 0보다 큰값에는 2, 작은 값은 -2
print('\n',np.where(arr>0,2,arr))   # o보다 큰값에는 2, 작으면 본래의 값
