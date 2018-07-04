"""Damn Paths"""
sys.path[0] = sys.path[0][0:-9]

from SET.graph import Graph


import pandas as pd
import tensorflow as tf
import numpy as np
import time
import sys


###########################
# IM too slow to get      #
# any reasonable results  #
# here yet :(             #
###########################

tf.enable_eager_execution()
train = pd.read_csv("/Users/jonval/.kaggle/competitions/digit-recognizer/train.csv")


train_input = np.array(train.drop(columns=["label"]))
train_label = np.array(train["label"])

###############################
# Network is Terribly Slow :( #
###############################

train_input = train_input[0:200]
train_label = train_label[0:200]
print("Input reading done...")

def one_hot(indexes):
    a = []
    for i in indexes:
        label = np.zeros(10)
        label[i] = 1

        a.append(label)

    return np.array(a)

EPOCHS   = 1
BATCH_SZ = 20

timestap = time.time()

G = Graph( 28 * 28, 10
         , size=2000
         , sleep_update=0.1
         , surges=3
         , c_dist=lambda: np.random.normal(loc=2, scale=1)
         , activation=tf.nn.tanh
         , output_activation=tf.nn.softmax
         , initializer=lambda: 0
         , loss=lambda labels, preds: tf.losses.softmax_cross_entropy(labels, preds)
         # , optimizer=tf.train.AdamOptimizer(0.01)
         , summarize=True
         , visualize=True)

print("Building Graph took {} seconds".format(time.time() - timestap))
print()

for i in range(EPOCHS):
    samples = len(train_label)
    batch   = BATCH_SZ
    
    while batch < samples:
        batchX = train_input[batch - BATCH_SZ: batch].reshape(-1, 28 * 28)
        batchY = train_label[batch - BATCH_SZ: batch]
        batchY = one_hot(batchY)

        batch += BATCH_SZ
        
        timestamp = time.time()
        G.train(batchX, batchY)
        print("Training Batch Took {}".format(time.time() - timestamp))

        BATCH = int(batch / BATCH_SZ)
        print("Batch {}/{}  ".format(BATCH, int(samples / BATCH_SZ)))

        timestamp = time.time()
        G.sleep()
        print("Sleeping Took {}".format(time.time() - timestamp))


preds = np.argmax(G.predict(train_input.reshape(-1, 28 * 28)), axis=1)

print(preds)
print(train_label)
print(sum(preds == train_label) / len(preds))
plt.show()


        








