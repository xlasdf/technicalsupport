# https://lge.smartlearn.io/asset-v1:POSTECH+DSC502+LGE901+type@asset+block/13_ANN_MNIST.html

# ANN with MNIST
# Import Library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#mnist dataset에서 data 가져오기
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ##### MNIST data set의 이해
# # train data
# print ("The training data set is:")
# print (mnist.train.images.shape) # images shape
# print (mnist.train.labels.shape) # images label 개수
# print('\n')
# # test data
# print ("The test data set is:")
# print (mnist.test.images.shape)
# print (mnist.test.labels.shape)
#
# # n번째 에 해당하는 train data 가져오기
# mnist.train.images[5]
# print(mnist.train.images[5].shape)
#
# #이미지 shape 바꾸기
# img = np.reshape(mnist.train.images[1],[28,28])
# img2 = mnist.train.images[5].reshape([28,28])
# img.shape
# img.shape
#
# # 3의 one_hot 인코딩 label 가져오기
# mnist.train.labels[1]
# print(mnist.train.labels[1])
#
# #argmax 사용해서 one_hot 인코딩된 label의 번지수 가져오기
# #0또는1 중 하나 이므로 1인 위치를 가져온다고 보면된다.
# np.argmax(mnist.train.labels[1])
#
# # 숫자 3 개 가져오기
# x, y = mnist.train.next_batch(3)
#
# print(x)
# print(y)
#
# plt.figure(figsize=(6,6))
# plt.imshow(img, 'gray')
# plt.xticks([])
# plt.yticks([])
# plt.show()
##
# 이미지 batch 해서 보이기
# train_x, train_y = mnist.train.next_batch(1)
# img = train_x[0,:].reshape([28,28])
#
# plt.figure(figsize=(6,6))
# plt.imshow(img, 'gray')
# plt.title("label : {}".format(np.argmax(train_y[0,:]))) # argmax사용해서 index 가져오기
# plt.xticks([])
# plt.yticks([])
# plt.show()



n_input = 28*28
n_hidden = 100
n_output = 10

weights = {
    'hidden' : tf.Variable(tf.random_normal([n_input, n_hidden], stddev =0.1)),
    'output' :tf.Variable(tf.random_normal([n_hidden, n_output], stddev =0.1))
}

biaes = {
    'hidden' : tf.Variable(tf.random_normal([n_hidden], stddev =0.1)),
    'output' :tf.Variable(tf.random_normal([n_output], stddev =0.1))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

# Model 정의
## Define Network

# y = w*x + b
def build_model(x, weights, biaes):
    hidden = tf.add(tf.matmul(x, weights['hidden']), biaes['hidden'])
    hidden = tf.nn.relu(hidden)

    output = tf.add(tf.matmul(hidden, weights['output']), biaes['output'])
    return output

#Define Loss funcion

pred = build_model(x, weights, biaes) # y_hat
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y) # 이값이 loss
loss = tf.reduce_mean(loss)

LR = 0.0001
optm = tf.train.AdamOptimizer(LR).minimize(loss)

n_batch = 50
n_iter = 5000
n_prt = 250

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_record_train = []
loss_record_test = []

for epoch in range(n_iter):
    train_x, train_y = mnist.train.next_batch(n_batch)
    sess.run(optm, feed_dict = {x: train_x, y : train_y})

    # 250번 마다 저장을 하자
    if epoch % n_prt == 0:
        test_x, test_y = mnist.test.next_batch(n_batch)
        c1 = sess.run(loss, feed_dict = {x: train_x, y : train_y}) # 250번 마다의 loss
        c2 = sess.run(loss, feed_dict = {x: test_x, y : test_y})
        loss_record_train.append(c1)
        loss_record_test.append(c2)
print("Iter : {}".format(epoch))
print("Cost : {}".format(c1))

#loss plot

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(loss_record_train))*n_prt,
        loss_record_train, label = 'training')
plt.plot(np.arange(len(loss_record_test))*n_prt,
        loss_record_test, label = 'testing')
plt.xlabel('iteration', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.legend(fontsize=12)
plt.ylim([0, np.max(loss_record_train)])

## Test Evaluate
test_x, test_y = mnist.test.next_batch(100)
my_pred = sess.run(pred, feed_dict={x:test_x}) #predict만 하므로 x 만 넣는다.
# 위 는 one_hot 인코딩으로 나오므로
my_pred = np.argmax(my_pred, axis = 1) #행방향 최대

labels = np.argmax(test_y, axis = 1)

accr = np.mean(np.equal(my_pred, labels)) # 두개 같은 것의 비율
print("Accuracy : {}".format(accr*100))

# 한개 가져와서 predict 하자
test_x, test_y = mnist.test.next_batch(1)

logits = sess.run(tf.nn.softmax(pred), feed_dict = {x : test_x}) # softmax를 취해서 확률 값으로
predict = np.argmax(logits, axis = 1)

print('Predtion : {}'.format(predict))
np.set_printoptions(precision = 2, suppress = True)
print('Probability : {}'.format(logits.ravel()))
plt.figure(figsize = (6,6))
plt.imshow(test_x.reshape(28,28), 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

## 위문제는 이미지로 바라보는 것이 더 좋은 method이다.
# 이는 CNN 으로 확장된다. 이는 Accuracy가 높아진다.
