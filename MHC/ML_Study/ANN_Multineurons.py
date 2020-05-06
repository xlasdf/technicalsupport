import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/12_ANN.html#4.1.-Multi-Layers

# training data gerneration
from sklearn.preprocessing import OneHotEncoder

m = 1000
x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1*x2-1)**2 + 2*x2 + 5

C1 = np.where(g >= 0)[0]
C0 = np.where(g < 0)[0]
N = C1.shape[0]
M = C0.shape[0]
m = N + M

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

train_X = np.vstack([X1, X0])
train_X = np.asmatrix(train_X)

train_y = np.vstack([np.ones([N,1]), np.zeros([M,1])])
ohe = OneHotEncoder(handle_unknown='ignore')
train_y = ohe.fit_transform(train_y).toarray()

n_input = 2
n_hidden = 4
n_output = 2

def build_model(x, weights, biases):
    hidden = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden = tf.nn.sigmoid(hidden)

    output = tf.add(tf.matmul(hidden, weights['output']), biases['output'])
    return output

weights = {
    'hidden' : tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 0.1)),
    'output' : tf.Variable(tf.random_normal([n_hidden, n_output], stddev = 0.1))
}

biases = {
    'hidden' : tf.Variable(tf.random_normal([n_hidden], stddev = 0.1)),
    'output' : tf.Variable(tf.random_normal([n_output], stddev = 0.1))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

pred = build_model(x, weights, biases)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y)
loss = tf.reduce_mean(loss)

LR = 0.01
optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

n_batch = 50
n_iter = 80000
n_prt = 250

# Training cycle
loss_record = []
for epoch in range(n_iter):
    sess.run(optm, feed_dict = {x: train_X,  y: train_y})
    if epoch % n_prt == 0:
        loss_record.append(sess.run(loss, feed_dict = {x: train_X,  y: train_y}))

w_hat = sess.run(weights)
b_hat = sess.run(biases)

# loss plots
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record))*n_prt, loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

x1p = np.arange(-5, 5, 0.01).reshape(-1, 1)
x2p = - w_hat['hidden'][0,0]/w_hat['hidden'][1,0]*x1p - b_hat['hidden'][0]/w_hat['hidden'][1,0]
x3p = - w_hat['hidden'][0,1]/w_hat['hidden'][1,1]*x1p - b_hat['hidden'][1]/w_hat['hidden'][1,1]
x4p = - w_hat['hidden'][0,2]/w_hat['hidden'][1,2]*x1p - b_hat['hidden'][2]/w_hat['hidden'][1,2]
x5p = - w_hat['hidden'][0,3]/w_hat['hidden'][1,3]*x1p - b_hat['hidden'][3]/w_hat['hidden'][1,3]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'k', linewidth = 3, label = '')
plt.plot(x1p, x3p, 'g', linewidth = 3, label = '')
plt.plot(x1p, x4p, 'm', linewidth = 3, label = '')
plt.plot(x1p, x5p, 'c', linewidth = 3, label = '')
plt.xlabel('$x_1$', fontsize = 15)
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.axis('equal')
plt.xlim([-5, 5])
plt.ylim([-4, 4])
plt.show()
