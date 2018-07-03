import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# tf.enable_eager_execution()

# def trainable_variable(shape):
    # initializer = tf.truncated_normal(shape, mean=0, stddev=0.3)
    # return tfe.Variable(initializer)


# x = [[2.]]
# y = tfe.Variable([[4.0]])
# m = tf.matmul(x, y)
# print("hello, {}".format(m))  # => "hello, [[4.]]"


# w1 = trainable_variable([10, 10])
# b1 = trainable_variable([10])

# w2 = trainable_variable([1, 10])
# b2 = trainable_variable([10])

# x = np.arange(10, dtype=np.float32).reshape(-1, 1)
# y = (np.arange(10, dtype=np.float32) * 2).reshape(-1, 1)
# xt = (np.arange(10, dtype=np.float32) + 10).reshape(-1, 1)

# optimizer = tf.train.AdamOptimizer(0.01)

# def propagate(x):
    # h1 = tf.matmul(w1, x)
    # h1 = tf.nn.tanh(tf.add(h1, b1))

    # h2 = tf.matmul(w2, h1)
    # h2 = tf.add(h2, b2)

    # return tf.reshape(h2, (-1, 1))

# for _ in range(1000):

    # error = lambda: tf.losses.mean_squared_error(y, propagate(x))
    # print(error())
    # optimizer.minimize(error)


# print(x)
# print(xt)

# print(propagate(x))
# print(propagate(xt))





