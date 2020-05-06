import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.Variable(0, dtype = tf.float32)

cost = w*w - 8*w +16
# cost1 = w**2 - 8*w + 16
# cost2 = tf.matmul(w, w) - 8*w + 16

LR = 0.05 # learning Rate
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init) # 여기 까지 하면 w에 0이할당된다.
    # tf.global_variables_initializer().run()
    optm = tf.train.GradientDescentOptimizer(LR).minimize(cost)

    for _ in range(300):
        sess.run(optm)
        W = sess.run(w)
    print(W)
