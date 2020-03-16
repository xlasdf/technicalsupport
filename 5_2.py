#Pandas 자료구조
#Data Frame
#표 같은 스프레드시트 형식의 자료 구조로 여러 개의 칼럼이 있는데 서로 다른종류의 값을
#담을 수 있다.
#DataFrame은 색인의 모양이 같은 Series 객체를 담고 있는 파이썬 사전으로 생각하면 편하다.

import pandas as pd
import numpy as np
data = {'State': ['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
        'year': [2000,2001,2002,2001,2002,2003],
        'pop': [1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)
print('\n',frame)
# dict Type으로 기입, key는 Column, value는 listtype으로 기입

print(frame['State'])
print(frame.State)  # 이와같이 접근가능

# print(frame.loc['three'])  #row 값을 선택하기 위해서는 loc 메서드를 이용,

# dataframe 생성자에서 사용가능한 입력 데이터
# 2차원 ndarray
# 배열, 리스트, 튜퓰의 사전
# numpy의 구조와 배열
# Series 사전
# 사전이나 Series의 리스트
# 리스트나 튜플의 리스트
# 다른 DataFrame

frame1 = pd.DataFrame(data, columns = ['year', 'State', 'pop', 'debt'],
                        index=['one','two','three','four',
                               'five','six'])

frame1['debt'] = 20   #debt을 채워넣는 법
frame1['debt'] = np.arange(6)   #0~5까지 나열


print('\n',frame1)  # NAN Not a Number == Null
print('\n',frame1.columns)
print('\n',frame1['State'])
print('\n',frame1.year)
print('\n',frame1.loc['two'])   #2행의 값을 추출한다. Series 형태로

print('\n',frame1.values)   #data 만 2차원 형태로 추출, ndarray 형태
