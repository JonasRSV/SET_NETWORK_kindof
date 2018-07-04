import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VizGraph(object):

    def __init__( self
                , trainable_variables
                , nodes
                , input_nodes
                , output_nodes
                , default_node_sz=300):
        self.tvs = trainable_variables
        self.nod = nodes
        self.ino = input_nodes
        self.ono = output_nodes
        self.dns = default_node_sz

    def draw(self):
        graph = nx.MultiGraph()

        node_colors = []
        node_sizes   = []
        edge_colors = []


        for node in self.nod:
            color = "black"

            if node in self.input_nodes:
                color = "green"

            if node in self.output_nodes:
                color = "red"

            node_colors.append(color)
            graph.add_node(node.key)

        for node in self.nod:
            node_w = 0
            for strength, con in node.connections.items():
                strength = np.float32(strength[0])

                node_w += abs(strength)

                graph.add_edge(node.key, con.key)
                edge_colors.append(np.tanh(strength))

            node_sizes.append(node_w)


        node_sizes = np.array(node_sizes)
        node_sizes = node_sizes / np.mean(node_sizes)
        node_sizes = node_sizes * self.dns

        nx.draw( graph
               , node_size=node_sizes
               , node_color=node_colors
               , edge_color=edge_colors
               , edge_vmin=-1
               , edge_vmax=1
               , alpha=0.8
               , edge_cmap=plt.get_cmap("seismic"))
        plt.pause(0.0001)

