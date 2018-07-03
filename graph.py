import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
from node import Node


class Graph(object):

    def __init__( self
                , inputs
                , outputs
                , size=10
                , sleep_update=0.05
                , surges=10
                , c_dist=lambda: np.random.normal(8, 2)
                , activation=tf.nn.tanh
                , initializer=lambda: 0
                , loss=tf.losses.mean_squared_error
                , optimizer=tf.train.GradientDescentOptimizer(0.01)
                , logdir="./summaries"):
        

        ####################################################################
        # Inputs: number of input nodes in the graph                       #
        # Outputs: number of output nodes in the graph                     #
        # Size: number of nodes in the graph                               #
        # Sleep Update: Number of Edges Updated Every Sleep Cycle          #
        # Surges: Number of surges through the network                     #
        # c_dist: distribution of connections per node   #
        # activation: activation function for each connection of each node #
        # initializer: activity initializer for each node                  #
        # popularities: How popular it is to connect to that node          #
        # nodes: All the nodes in the graph                                #
        # trainable_variables: variables that is trained                   #
        # loss: Loss used to evaluate result                               #
        # Optimizer: Function to minimize the loss                         #
        # Logdir: Directory to store summaries
        ####################################################################

        self.inputs      = inputs
        self.outputs     = outputs
        self.size        = size
        self.surges      = surges
        self.cd          = c_dist
        self.activation  = activation
        self.initializer = initializer
        self.loss        = loss
        self.optimizer   = optimizer

        #################
        #               #
        # For Summaries #
        #               #
        #################

        self.summaries   =  tf.contrib.summary.create_file_writer(logdir)
        self.summaries.set_as_default()

        self.global_step = tf.train.get_or_create_global_step()

        ###################
        #                 #
        # Store the Graph #
        #                 #
        ###################
        self.nodes               = [None] * self.size
        self.trainable_variables = []

        self.initialize()

        self.summarize()

    def summarize(self):
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):

            for variable in self.trainable_variables:
                name = "/".join(variable.name.split("/")[0:2])
                
                print(name)
                tf.contrib.summary.scalar(name, variable)

        self.global_step.assign_add(1)

    def initialize(self):
        for i in range(self.size):
            c             = int(abs(self.cd()))
            self.nodes[i] = Node( i
                                , self
                                , c
                                , activation=self.activation
                                , initial=self.initializer())

        for i in range(self.size):
            self.nodes[i].initialize()

        #################################
        #                               #
        # Select Input and Output nodes #
        #                               #
        #################################

        self.input_nodes  = np.random.choice( self.nodes[:int(len(self.nodes) / 2)]
                                            , self.inputs)
        self.output_nodes = np.random.choice( self.nodes[int(len(self.nodes) / 2):]
                                            , self.outputs)

        return None

    def __call__(self, inputs):
        for I, N in zip(inputs, self.input_nodes):
            N.surge(tf.constant(I))

        for _ in range(self.surges):
            for node in self.nodes:
                node()

        output = []
        for i, node in enumerate(self.output_nodes):
            output.append(node.activity)
            
            """Reset Node."""
            node.cool()

        return output

    def train_single_cycle( self
                          , inputs
                          , labels):

        """
        Train on one cycle.
        Not sure how to incorporate batched training yet.
        """

        with tfe.GradientTape(persistent=True) as gradients:
            """Surge network."""
            outputs = self(inputs)
            loss = self.loss(labels.reshape(-1, 1), outputs)

        var_grads = gradients.gradient(loss, self.trainable_variables)

        """None Gradients, I assume they did not affect outcome at all."""
        var_grads = [grad if grad is not None else tf.constant([0], dtype=tf.float32)
                        for grad in var_grads]

        
        """Avoid Exploding Gradients."""
        var_grads = tf.clip_by_value(var_grads, -100, 100)
        self.optimizer.apply_gradients(zip(var_grads, self.trainable_variables))

        self.summarize()

        return loss



if __name__ == "__main__":
    tf.enable_eager_execution()
    g = Graph( 5, 5
             , size=50
             , surges=4
             , c_dist=lambda: 1)

    inputs  = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    outputs = np.array([1, 0, -1, -2, -3], dtype=np.float32)

    for _ in range(100):
        print("Loss", g.train_single_cycle(inputs, outputs))

    # ts = time.time()
    # print(g(inputs))
    # print(time.time() - ts)

    # print(len(g.trainable_variables))

