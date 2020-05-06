#Linear Regression for Tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data points in column vector [input, output]
train_x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
train_y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)
# 13x1
# y = wx + b

m = train_x.shape[0]

LR = 0.001
n_iter = 300

x = tf.placeholder(tf.float32, [m,1])
y = tf.placeholder(tf.float32, [m,1])

w = tf.Variable([[0]], dtype = tf.float32)
b = tf.Variable([[0]], dtype = tf.float32)

#y_pred = tf.matmul(x, w) + b
y_pred = tf.add(tf.matmul(x, w),b) # y_pred => y^,hat
# 행렬연산위해
loss = tf.square(y_pred-y)  # vectorized 된 값이므로
loss = tf.reduce_mean(loss) # 평균으로 한가지 값으로 추린다.

optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_record = []

for epoch in range(n_iter):
    _, c = sess.run([optm, loss], feed_dict = {x:train_x, y:train_y})
    loss_record.append(c)

w_val = sess.run(w)
b_val = sess.run(b)
sess.close()

# optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(n_iter):
#         sess.run(optm, feed_dict = {x: train_x, y: train_y})
#     w_val = sess.run(w)
#     b_val = sess.run(b)



print(w_val,'\n')
print(b_val,'\n')

xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp = w_val*xp + b_val


plt.figure(figsize = (10,8))
plt.plot(loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

plt.figure(figsize = (10,8))
plt.plot(train_x, train_y, 'ko')
plt.plot(xp, yp, 'r')
plt.title('Data', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()
