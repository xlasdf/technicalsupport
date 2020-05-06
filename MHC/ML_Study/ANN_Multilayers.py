import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/12_ANN.html#4.1.-Multi-Layers

# Multi-Layers

# training data gerneration
from sklearn.preprocessing import OneHotEncoder
m = 1000
x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1-1)**2 + 2*x2 + 5

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
n_hidden = 2
n_output = 2

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

def build_model(x, weights, biases):
    hidden = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden = tf.nn.sigmoid(hidden)

    output = tf.add(tf.matmul(hidden, weights['output']), biases['output'])
    return output

pred = build_model(x, weights, biases)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y)
loss = tf.reduce_mean(loss)

LR = 0.01
optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

n_batch = 50
n_iter = 50000
n_prt = 250

loss_record = []
for epoch in range(n_iter):
    sess.run(optm, feed_dict = {x: train_X,  y: train_y})
    if epoch % n_prt == 0:
        loss_record.append(sess.run(loss, feed_dict = {x: train_X,  y: train_y}))

w_hat = sess.run(weights)
b_hat = sess.run(biases)

# loss plot
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record))*n_prt, loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)


H = train_X*w_hat['hidden'] + b_hat['hidden']
H = 1/(1 + np.exp(-H))

# plt.figure(figsize=(10, 8))
# plt.plot(H[0:N,0], H[0:N,1], 'ro', alpha = 0.4, label = 'C1')
# plt.plot(H[N:m,0], H[N:m,1], 'bo', alpha = 0.4, label = 'C0')
# plt.xlabel('$z_1$', fontsize = 15)
# plt.ylabel('$z_2$', fontsize = 15)
# plt.legend(loc = 1, fontsize = 15)
# plt.axis('equal')
# plt.xlim([0, 1])
# plt.ylim([0, 1])

x1p = np.arange(0, 1, 0.01).reshape(-1, 1)
x2p = - w_hat['output'][0,0]/w_hat['output'][1,0]*x1p - b_hat['output'][0]/w_hat['output'][1,0]
x3p = - w_hat['output'][0,1]/w_hat['output'][1,1]*x1p - b_hat['output'][1]/w_hat['output'][1,1]

# hidden layer plot
plt.figure(figsize=(10, 8))
plt.plot(H[0:N,0], H[0:N,1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(H[N:m,0], H[N:m,1], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'k', linewidth = 3, label = '')
plt.plot(x1p, x3p, 'g', linewidth = 3, label = '')
plt.xlabel('$z_1$', fontsize = 15)
plt.ylabel('$z_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])

x1p = np.arange(-5, 5, 0.01).reshape(-1, 1)
x2p = - w_hat['hidden'][0,0]/w_hat['hidden'][1,0]*x1p - b_hat['hidden'][0]/w_hat['hidden'][1,0]
x3p = - w_hat['hidden'][0,1]/w_hat['hidden'][1,1]*x1p - b_hat['hidden'][1]/w_hat['hidden'][1,1]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'k', linewidth = 3, label = '')
plt.plot(x1p, x3p, 'g', linewidth = 3, label = '')
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.axis('equal')
plt.xlim([-5, 5])
plt.ylim([-4, 4])
plt.show()
