import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
from node import Node
from vizgraph import VizGraph


class Graph(VizGraph):

    def __init__( self
                , inputs
                , outputs
                , size=10
                , sleep_update=0.1
                , surges=10
                , c_dist=lambda: np.random.normal(8, 2)
                , activation=tf.nn.relu
                , output_activation=tf.nn.relu 
                , initializer=lambda: 0
                , loss=tf.losses.mean_squared_error
                , optimizer=tf.train.GradientDescentOptimizer(0.01)
                , logdir="./summaries"
                , summarize=True
                , visualize=True):
        

        ####################################################################
        # Inputs: number of input nodes in the graph                       #
        # Outputs: number of output nodes in the graph                     #
        # Size: number of nodes in the graph                               #
        # Sleep Update: Number of Edges Updated Every Sleep Cycle          #
        # Surges: Number of surges through the network                     #
        # c_dist: distribution of connections per node                     #    
        # activation: activation function for each connection of each node #
        # output_activation: activation for final outputs                  #
        # initializer: activity initializer for each node                  #
        # popularities: How popular it is to connect to that node          #
        # nodes: All the nodes in the graph                                #
        # trainable_variables: variables that is trained                   #
        # loss: Loss used to evaluate result                               #
        # Optimizer: Function to minimize the loss                         #
        # Logdir: Directory to store summaries                             #
        ####################################################################

        self.inputs       = inputs
        self.outputs      = outputs
        self.size         = size
        self.sleep_update = sleep_update
        self.surges       = surges
        self.cd           = c_dist
        self.activation   = activation
        self.initializer  = initializer
        self.loss         = loss
        self.optimizer    = optimizer

        #################
        #               #
        # For Summaries #
        #               #
        #################

        self.summarize   = summarize
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

        self.__initialize()

        self.visualize = visualize
        if self.visualize:
            VizGraph.__init__( self
                             , self.trainable_variables
                             , self.nodes
                             , self.input_nodes
                             , self.output_nodes)

            self.draw()

    def __summarize_graph(self, **kwargs):
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):

            ######################################
            #            Wierd Bug!              #
            # Adding Summary Once Does not Work  #
            # Adding Summar Twice Does the Trick #
            ######################################
            for _ in range(2):
                tf.contrib.summary.scalar("loss", kwargs["loss"])

            # for variable in self.trainable_variables:
                # tf.contrib.summary.scalar(str(kwargs["loss"]), kwargs["loss"])
                # tf.contrib.summary.scalar(name, variable)

        self.global_step.assign_add(1)

    def __initialize(self):
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

    def __predict(self, inputs):
        assert 2 == len(inputs.shape)

        batch_dim, input_dim = inputs.shape

        assert input_dim == len(self.input_nodes)

        #############################################
        # Initialize all nodes to vectorized dim    #
        # Otherwise will be problematic when taking #
        # Gradients                                 #
        #############################################

        for node in self.nodes:
            node.set(tf.constant(
                np.array([self.initializer()] * batch_dim), 
                    dtype=tf.float32))


        ####################
        # Vectorize Inputs #
        ####################
        vector = inputs.T

        for i in range(input_dim):
            self.input_nodes[i]\
                .set(tf.constant(vector[i], dtype=tf.float32))


        for _ in range(self.surges):
            for node in self.nodes:
                node()

        output = list(
                    map(lambda node: self.output_activation(node.activity)
                        , self.output_nodes))

        for node in self.nodes:
            node.cool()

        return output

    def predict(self, inputs):
        outputs = self.__predict(inputs)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs)

        return np.array(outputs)

    def train( self
             , inputs
             , labels):

        assert 2 == len(inputs.shape)
        assert 2 == len(labels.shape)

        with tfe.GradientTape(persistent=True) as gradients:
            """Surge through network."""
            outputs = self.__predict(inputs)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs)
            loss = self.loss(labels, outputs)

        var_grads = gradients.gradient(loss, self.trainable_variables)

        """None Gradients, I assume they did not affect outcome at all."""
        var_grads = [grad if grad is not None else tf.constant([0], dtype=tf.float32)
                        for grad in var_grads]

        """Avoid Exploding Gradients."""
        var_grads = tf.clip_by_value(var_grads, -10, 10)
        self.optimizer.apply_gradients(zip(var_grads, self.trainable_variables))

        mean_loss = loss / len(inputs)

        if self.summarize:
            self.__summarize_graph(loss=mean_loss)

        return mean_loss

    def sleep(self):
        ############################################
        # Reconnect All the Weakest connections;   #
        # Essentially destroying them and creating #
        # new ones.                                #
        ############################################

        abs_con = list(map(lambda con: float(tf.abs(con)), self.trainable_variables))
        var_str = list(zip(abs_con, self.trainable_variables))
        var_str.sort(key=lambda x: x[0])

        reconnections = int(self.sleep_update * len(self.trainable_variables))
        reconnected   = 0

        for _, variable in var_str:
            if reconnected >= reconnections:
                break

            node, id = variable.name.split("/")[0:2]

            """I don't want to mix with the bias variables."""
            if id == "bias":
                continue

            self.nodes[int(node)].reconnect(variable)
            reconnected += 1

        if self.visualize:
            self.draw()

        return None



if __name__ == "__main__":
    tf.enable_eager_execution()
    g = Graph( 5, 5
             , size=100
             , surges=5)

    # g.sleep()
    # inputs  = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    # outputs = np.array([1, 4, 6, 8, 10], dtype=np.float32)

    # inputs  = np.array([inputs, inputs, inputs])
    # outputs = np.array([outputs, outputs, outputs])

    # for _ in range(10):
        # g.train(inputs, outputs)

    time.sleep(10)

    # print(g.predict(inputs))


