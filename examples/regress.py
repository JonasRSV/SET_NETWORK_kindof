import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

"""Damn Paths."""
sys.path[0] = sys.path[0][0:-9]

from graph import Graph

tf.enable_eager_execution()


NOISE_SCALAR = 2
EPOCHS = 50
BATCHSZ = 10

Xtraining = np.arange(100)
Ytraining = np.arange(100) + (np.random.rand() - 0.5) * NOISE_SCALAR


G = Graph(1, 1
         , size=40
         , sleep_update=0.1
         , surges=1
         , c_dist=lambda: np.random.normal(4, 1)
         , activation=tf.nn.relu
         , output_activation=lambda x: x
         , loss=tf.losses.mean_squared_error
         , optimizer=tf.train.GradientDescentOptimizer(0.01)
         , summarize=True
         , visualize=True
         , default_node_sz=100)



# print(Xtraining.reshape(-1, 1))
# print(G.predict(Xtraining.reshape(-1, 1)))
# print(G.output_nodes[0].activity)

for i in range(EPOCHS):

    for batch in range(int(len(Xtraining) / BATCHSZ)):
        b = batch * BATCHSZ

        Xbatch = Xtraining[b:b + BATCHSZ].reshape(-1, 1)
        Ybatch = Ytraining[b:b + BATCHSZ].reshape(-1, 1)

        G.train(Xbatch, Ybatch)

    G.sleep()

    print(i)


print(G.output_nodes[0].activity)
print(Xtraining)
print(G.predict(Xtraining.reshape(-1, 1)))



plt.show()
