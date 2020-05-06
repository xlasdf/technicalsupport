import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# embaded funcion in tensorflow

# datat generation

m = 1000
true_w = np.array([[-6], [2], [1]])
train_X = np.hstack([np.ones([m,1]), 5*np.random.rand(m,1), 4*np.random.rand(m,1)])

true_w = np.asmatrix(true_w)
train_X = np.asmatrix(train_X)
train_y = 1/(1 + np.exp(-train_X*true_w)) > 0.5

C1 = np.where(train_y == True)[0]
C0 = np.where(train_y == False)[0]

train_y = np.empty([m,1])
train_y[C1] = 1
train_y[C0] = 0

LR = 0.05
n_iter = 15000

# train_x와 train_y를 연산할때 마다 placeholder에 넣는다.
X = tf.placeholder(tf.float32, [m,3])
y = tf.placeholder(tf.float32, [m,1])
w = tf.Variable([[0],[0],[0]],dtype=tf.float32)

y_pred = tf.sigmoid(tf.matmul(X,w))
loss = - y*tf.log(y_pred) - (1-y)*tf.log(1-y_pred) #minimize하기위해 -
loss = tf.reduce_mean(loss)

optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)
loss_record = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_iter):
        _, c = sess.run([optm, loss], feed_dict = {X: train_X, y: train_y})
        loss_record.append(c)

    w_hat = sess.run(w)
print(w_hat)

xp = np.arange(0, 4, 0.01).reshape(-1, 1)

yp = - w_hat[1,0]/w_hat[2,0]*xp - w_hat[0,0]/w_hat[2,0]

plt.figure(figsize = (10,8))
plt.plot(loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

plt.figure(figsize = (10,8))
plt.plot(train_X[C1,1], train_X[C1,2], 'ro', alpha = 0.3, label='C1')
plt.plot(train_X[C0,1], train_X[C0,2], 'bo', alpha = 0.3, label='C0')
plt.plot(xp, yp, 'g', linewidth = 3, label = 'Logistic Regression')
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.ylim([0,4])
plt.show()


### method 2
# X = tf.placeholder(tf.float32, [m,3])
# y = tf.placeholder(tf.float32, [m,1])
# w = tf.Variable(tf.random_normal([3,1]), dtype = tf.float32)
#
# y_pred = tf.matmul(X,w)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred, labels = y) ## tensorflow sigmoid funcion
# loss = tf.reduce_mean(loss)
#
# optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(n_iter):
#         sess.run(optm, feed_dict = {X: train_X, y: train_y})
#
#     w_hat = sess.run(w)
#
# print(w_hat)
#
# xp = np.arange(0, 4, 0.01).reshape(-1, 1)
# yp = - w_hat[1,0]/w_hat[2,0]*xp - w_hat[0,0]/w_hat[2,0]
