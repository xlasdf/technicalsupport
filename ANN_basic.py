## Artificial Neural Networks

# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/12_ANN.html
# https://www.youtube.com/embed/3liCbRZPrZA?rel=0

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


## logistic Regression이 여러개가 있는 것과 같다.
#training data gerneration
m = 1000
x1 = 8*np.random.rand(m, 1)
x2 = 7*np.random.rand(m, 1) - 4

g = 0.8*x1 + x2 - 3

C1 = np.where(g >= 0)[0]
C0 = np.where(g < 0)[0]
N = C1.shape[0]
M = C0.shape[0]
m = N + M

X1 = np.hstack([np.ones([N,1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([M,1]), x1[C0], x2[C0]])

train_X = np.vstack([X1, X0])
train_y = np.vstack([np.ones([N,1]), -np.ones([M,1])])

train_X = np.asmatrix(train_X)
train_y = np.asmatrix(train_y)

train_y = np.vstack([np.ones([N,1]), np.zeros([M,1])]) # zeros 로 변경
train_y = np.asmatrix(train_y)

LR = 0.05
n_iter = 15000

x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.random_normal([3,1]))

y_pred = tf.matmul(x,w)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred, labels = y)
loss = tf.reduce_mean(loss)

optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_iter):
        sess.run(optm, feed_dict = {x: train_X, y:train_y})

    w_hat = sess.run(w)

x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[1,0]/w_hat[2,0]*x1p - w_hat[0,0]/w_hat[2,0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'g', linewidth = 3, label = '')
plt.legend(loc = 1, fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.show()
