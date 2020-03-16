# # 난수 생성
# numpy.random 모듈은 다양한 종류의 확률분포로부터 효과적으로 표본값을 생성.
# 예를 들어 normal을 사용하여 표준 정규분포로 부터 4*4 크기의 표본을 생성할 수 있다.
# samples = np.random.normal(size=(4,4))
# numpy.random 모듈은 파이썬 내장random 모듈에 비해 더효율적으로 값을 생성하여
# 파이썬 내장모듈보다 수십배 이상 빠른 속도를 자랑한다.


# numpy.random 모듈 함수

# seed   난수발생기의 seed를 지정
# permutation   임의의 순열을 반환
# shuffle   리스트나 배열의 순서를 뒤섞는다
# rand   균등분포에서 표본을 추출
# randint  주어진 최소/최대범위 안에서 임의의 난수를추출
# randn 표준편차가 1이고 평균값이 0인 정규분포에서 표본을 추출
# binomial   이항분포에서 표본추출
# normal  정규분포(가우시안)에서 표본추출
# beta  베타분포에서 표본추출
# chisquare   카이제곱 분포에서 표본추출
# gamma 감마분포에서 표본 추출
# uniform 균등(0,1)에서 표본추출

import numpy as np

np.random.seed(12345)

data = np.random.rand(3,3) #3x3 균등 분포
data2 = np.random.randint(0,3) #0~3 3미만 정수값 1개 출력
data3 = np.random.randn(3,3) #3x3 정규분포
data4 = np.random.normal(size=(3,3)) #3x3 정규분포


print('rand:\n', data)
print('\nrandint:\n', data2)
print('\nrandn:\n', data3)
print('\nnormal:\n', data4)
