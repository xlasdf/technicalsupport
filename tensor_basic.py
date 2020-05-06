import tensorflow as tf
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/12_ML_with_tf.html

# CPU 경고 메시지 무시하기, 넌더 빨리 연산할수 있어!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1,2,3])
b = tf.constant(4, shape=[1,3])

A = a+b
B = a*b
print('Open Session EX')
sess = tf.Session() # session 열기
out1 = sess.run(A)
out2 = sess.run(B)
print(out1,'\n')
print(out2,'\n')
sess.close()

result = tf.multiply(a,b)

with tf.Session() as sess:
    output = sess.run(result)
    print(output,'\n') # with를 사용하면 연산후 자동으로 session close 됨

# tf.Variable

x1 = tf.Variable([1, 1]) # iterative 할건데 최초값설정
x2 = tf.Variable([2, 1])
y = x1+x2
#session 열어야 값이 나옴
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run(x2)
sess.run(y)
sess.close()

# tf.placeholder , 공간을 할당하고 필요시마다 원하는 값 넣는다.

x = tf.placeholder(tf.float32, shape=[2,2])

print('Placeholder EX')
sess = tf.Session()

arr_x = sess.run(x, feed_dict = {x:[[1,2], [3,4]]})
print(arr_x,'\n')

a = tf.placeholder(tf.float32, shape=[2])
b = tf.placeholder(tf.float32, shape=[2])

y = a + b

output = sess.run(y, feed_dict = {a: [1,2], b:[2,2]})

print(output,'\n')


##  Tensor Manipulation
print('Tensor Manipulation EX')
x1 = tf.constant(1, shape = [3]) # 1,1,1
x2 = tf.constant(2, shape = [3]) # 2,2,2
# output = tf.add(x1,x2)
output = x1 + x2

with tf.Session() as sess:
    result = sess.run(output)
    print(result,'\n')

a1 = tf.constant(1, shape = [2,3])
b1 = tf.constant(2, shape = [2,3])

output1 = tf.add(a1, b1)

with tf.Session() as sess:
    result = sess.run(output1)
    print(result,'\n')


###Multiplying Matrices
print('Multiplying Matrices EX' )
x1 = tf.constant([[1,2],
                  [3,4]])
x2 = tf.constant([[2],
                  [3]])

output2 = tf.matmul(x1, x2) # 행렬곱에서는 matmul 사용해야함

with tf.Session() as sess:
    result = sess.run(output2)
    print(result,'\n')

output3 = x1*x2 # 이는 각각 원소별 곱이다.

with tf.Session() as sess:
    result = sess.run(output3)
    print(result,'\n')

###Reshape
print('Reshape EX')
# x = [1, 2, 3, 4, 5, 6, 7, 8]
x = list(range(1,9))

x_re = tf.reshape(x, [4,2])

sess = tf.Session()
re_Data = sess.run(x_re)
print(re_Data,'\n')

x_re1 = tf.reshape(x, [2,4])
re_Data1 = sess.run(x_re1)
print(re_Data1,'\n')

# 행 or 열 shape 주어지면 알아서 결정해준다.
x_re2 = tf.reshape(x, [2,-1]) # 앞의 shape을 주어지면 뒤에 -1 해도 알아서 shape 결정해준다.
re_Data2 = sess.run(x_re2)
print(re_Data2,'\n')

x_re3 = tf.reshape(x, [-1,1]) # 뒤의 shape을 주어지면 뒤에 -1 해도 알아서 shape 결정해준다.
re_Data3 = sess.run(x_re3)
print(re_Data3,'\n') # like Transpose
