import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import sys

def trainable_variable(shape):
    initializer = tf.truncated_normal(shape, mean=0, stddev=0.3, dtype=tf.float32)
    return tfe.Variable(initializer, dtype=tf.float32)

class Node(object):

    def __init__(self, key, graph, cardinality, activation=tf.nn.relu, initial=0.0):

        #####################################################
        # Key: Unique Identifier for this node              #
        # Graph: The entire graph                           #
        # Cardinality: Cardinality of Connections           #
        # Activity: The activity of the node                #
        # Initial: The initial activation of the node       #
        # Activation: The non-linear activation of the node #
        # connections: Variables connected to nodes         #
        # Bias: The Bias of this Node                       #
        # Trainable Variables: --**--                       #
        #####################################################

        self.key         = key
        self.graph       = graph
        self.cardinality = cardinality
        self.activity    = tf.constant([initial], dtype=tf.float32)
        self.activation  = activation

        self.connections = {}
        with tf.variable_scope(str(self.key)):
            with tf.variable_scope("bias"):
                self.bias = trainable_variable([1])

        self.trainable_variables = [self.bias]

    def initialize(self):
        initial_con = np.random.choice(self.graph.nodes, size=self.cardinality)

        with tf.variable_scope(str(self.key)):
            for i in range(self.cardinality):
                with tf.variable_scope(str(i)):
                    variable_w = trainable_variable([1])
                    self.connections[variable_w] = initial_con[i]


        self.trainable_variables.extend(list(self.connections.keys()))
        self.graph.trainable_variables.extend(self.trainable_variables)

    def reconnect(self, variable):
        if variable not in self.connections:
            raise Exception(  "Tried to reconnect variable from node it was "
                            + "Not connected to; Variable {}, Node {} contains {}"
                            .format(variable, self, list(self.connections.keys())))

        del self.connections[variable]

        new_con = np.random.choice(self.graph.nodes)
        self.connections[variable] = new_con

    def surge(self, value):
        self.activity = self.activity + value

    def set(self, value):
        """https://stackoverflow.com/questions/12543837/python-iterating-over-list-vs-over-dict-items-efficiency"""
        self.view_cache = list(self.connections.items())
        self.activity = value

    def __call__(self):
        for weigth, connection in self.view_cache:
            connection.surge(self.activation(tf.multiply(self.activity, weigth) + self.bias))



