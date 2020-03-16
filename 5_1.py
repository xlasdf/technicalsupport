#pandas 자료구조
#pandas 소개 및 Series

# pandas는 고수준의 자료구조와 파이썬을 통한빠르고 쉬운 데이터 분석도구를 포함
# 명시적으로 축의 이름에 따라 데이터를 정렬할 수 있는 자료구조를 제공한다
# 통합된 시계열 데이터 처리 기능을 제공
# 시계열 데이터와 비시계열 데이터를 함께 다룰 수 있는 통합 자료구조를 제공
# 누락된 데이터 유연하게 처리 가능
# SQL과 같은 일반 데이터 베이스처럼 데이터를 합치고 관계 연산을 수행할 수 있다.


# Series
# 일련의 객체를 담을 수 있는 1차원 배열 같은 자료구조이다.
# 색인이라고 하는 배열 데이터에 연관된 이름을 가지고 있다.
# Series 객체의 문자열 표현은 왼쪽에 색인을 보여주고 오른쪽에 해당색의 값을 보여준다.

# Index   data   #data는 numpy의 array로
# 1       'A'
# 2       'B'
# 3       'C'
# 4       'D'
# 5       'E'

# dict와 같이 접근 가능

import pandas as pd
import numpy as np

# obj = pd.Series([4,7,-5,3])
# print('\n',obj)
# print('\n',obj.values)
# print('\n',obj.index)

obj2 = pd.Series([4,7,-5,3], index = ['d','b','a','c']) #index직접지정
print('\n',obj2)

# 불리언 배열을 사용해서 값을 걸러낼수 있다
# numpy 배열연산을 수행해도 색인값 연결은 유지됨.

print(obj2[obj2>0])
print('\n',obj2 * 2)
print('\n',np.exp(obj2))
