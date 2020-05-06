import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# logistic Regression이 여러개가 있는 것과 같다.
#training data gerneration
# Logistic Regression in a Form of Neural Network

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

### training set, weights method

# define input and output size
n_input = 3 # 1 , x1, x2
n_output = 1  # y 한개

# define weights as a dictionary
weights = {
    'output' : tf.Variable(tf.random_normal([n_input, n_output], stddev = 0.1))
}

# define placeholders for train_x and train_y
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

# define network architecture
def build_model(x, weights):
    output = tf.matmul(x, weights['output'])
    return output

# define loss
pred = build_model(x, weights)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels = y)
loss = tf.reduce_mean(loss)
LR = 0.05
optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

n_batch = 50     # Batch size
n_iter = 15000   # Learning iteration
n_prt = 250      # Print cycle

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# training or learning

loss_record = []
for epoch in range(n_iter):
    sess.run(optm, feed_dict = {x: train_X,  y: train_y})
    if epoch % n_prt == 0:
        loss_record.append(sess.run(loss, feed_dict = {x: train_X,  y: train_y}))

w_hat = sess.run(weights['output'])

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record))*n_prt, loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[1,0]/w_hat[2,0]*x1p - w_hat[0,0]/w_hat[2,0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'g', linewidth = 3, label = '')
plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.show()

### Weights and Bias method
### In a neural network, weights and biases are typically separated.
'''
n_input = 2
n_output = 1

train_X = train_X[:,1:3]
# define network
def build_model(x, weights, biases):
    output = tf.add(tf.matmul(x, weights['output']), biases['output'])
    return output

weights = {
    'output' : tf.Variable(tf.random_normal([n_input, n_output], stddev = 0.1))
}

biases = {
    'output' : tf.Variable(tf.random_normal([n_output], stddev = 0.1))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

pred = build_model(x, weights, biases)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
loss = tf.reduce_mean(loss)

LR = 0.05
optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_batch = 50
n_iter = 15000
n_prt = 250

loss_record = []
for epoch in range(n_iter):
    sess.run(optm, feed_dict = {x: train_X,  y: train_y})
    if epoch % n_prt == 0:
        loss_record.append(sess.run(loss, feed_dict = {x: train_X,  y: train_y}))

w_hat = sess.run(weights['output'])
b_hat = sess.run(biases['output'])

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record))*n_prt, loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[0,0]/w_hat[1,0]*x1p - b_hat[0]/w_hat[1,0] # 위와 다르게 bias 가 계산된걸 볼수 있다.

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'g', linewidth = 3, label = '')
plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.show()
'''


### One-hot Encoding
### One-hot encoding is a conventional practice for a multi-class classification
'''
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
train_y = ohe.fit_transform(train_y).toarray()
print(train_y)

n_input = 2
n_output = 2

weights = {
    'output' : tf.Variable(tf.random_normal([n_input, n_output], stddev = 0.1))
}

biases = {
    'output' : tf.Variable(tf.random_normal([n_output], stddev = 0.1))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

pred = build_model(x, weights, biases)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
loss = tf.reduce_mean(loss)

LR = 0.05
optm = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_batch = 50
n_iter = 15000
n_prt = 250

loss_record = []
for epoch in range(n_iter):
    sess.run(optm, feed_dict = {x: train_X,  y: train_y})
    if epoch % n_prt == 0:
        loss_record.append(sess.run(loss, feed_dict = {x: train_X,  y: train_y}))

w_hat = sess.run(weights['output'])
b_hat = sess.run(biases['output'])

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record))*n_prt, loss_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('loss', fontsize = 15)

print(w_hat)

x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[0,0]/w_hat[1,0]*x1p - b_hat[0]/w_hat[1,0]
x3p = - w_hat[0,1]/w_hat[1,1]*x1p - b_hat[1]/w_hat[1,1]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'k', linewidth = 3, label = '')
plt.plot(x1p, x3p, 'g', linewidth = 3, label = '')
plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.show()
'''
