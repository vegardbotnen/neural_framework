import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class Neuron:
    def __init__(self, identifier):
        self.identifier = identifier
        self.excitement = 0.0
        self.history = deque([0.0]*20,20)
        self.activation_function = lambda x: x

    def excite(self, value):
        self.excitement += value

    def get_activation(self, t):
        return self.history[t]

    def step(self):
        activation = self.activation_function(self.excitement)
        self.excitement = 0.0
        self.history.appendleft(activation)

    def __repr__(self):
        return "Neuron: " + str(self.identifier)


class Brain:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_neuron(self):
        identifier = len(self.graph.nodes)
        new_neuron = Neuron(identifier)
        self.graph.add_node(identifier, neuron=new_neuron)
        return identifier

    def add_synapse(self, from_id, to_id, weight, delay):
        self.graph.add_edge(from_id, to_id, weight=weight, delay=delay)

    def excite_neuron(self, identifier, value):
        neuron = self.graph.nodes[identifier]["neuron"]
        neuron.excite(value)

    def get_activation(self, neuron_ids):
        activation = []
        for neuron_id in neuron_ids:
            if neuron_id in self.graph.nodes:
                neuron = self.graph.nodes[neuron_id]["neuron"]
                activation.append(neuron.get_activation(t=0))
        return activation

    def step(self):
        # calculate excitement for each neuron
        for post_neuron_id in self.graph.nodes:
            post_neuron = self.graph.nodes[post_neuron_id]["neuron"]
            excitement = 0.0
            for pre_neuron_id in self.graph.predecessors(post_neuron_id):
                synapse = self.graph[pre_neuron_id][post_neuron_id]
                weight, delay = synapse["weight"], synapse["delay"]
                pre_neuron = self.graph.nodes[pre_neuron_id]["neuron"]
                activation = pre_neuron.get_activation(delay-1)
                print(f"Pre id: {pre_neuron_id} -> ({delay}) ->  post id: {post_neuron_id} -- Activation: {activation}")
                excitement += activation * weight
            post_neuron.excite(excitement)

        # calculate activation for each neuron
        for neuron_id in self.graph.nodes:
            neuron = self.graph.nodes[neuron_id]["neuron"]
            neuron.step()

    def show(self):
        pos = nx.drawing.layout.random_layout(self.graph)
        print(pos)
        nx.draw_networkx(self.graph, pos=pos, with_labels=True)
        # nx.draw_networkx_edge_labels(self.graph, pos=pos)
        plt.show()


# layout
def fully_connected(shape):
    fc = Brain()
    prev_layer = []
    tmp_layer = []

    # create input neurons
    for _ in range(shape[0]):
        neuron_id = fc.add_neuron()
        prev_layer.append(neuron_id)
        print(f"New input neuron: {neuron_id}")

    # create fully connected hidden- and output layers
    for n_neurons in shape[1:]:
        for _ in range(n_neurons):
            post_neuron_id = fc.add_neuron()
            tmp_layer.append(post_neuron_id)
            print(f"New other neuron: {post_neuron_id}")

            # create synapses from alle pre-layer neurons
            for pre_neuron_id in prev_layer:
                print(f"Synapse from {pre_neuron_id} -> {post_neuron_id}")
                weight, delay = 1.0, 1.0
                fc.add_synapse(pre_neuron_id, post_neuron_id, weight, delay)

        # switch to next layer
        prev_layer = tmp_layer

    # return fully-connected
    return fc


