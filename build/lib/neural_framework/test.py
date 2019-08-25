import unittest
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
from brain import Brain, fully_connected

class TestStringMethods(unittest.TestCase):

    def test_create_brain(self):
        brain = Brain()
        self.assertEqual(0, brain.num_neurons)
        n1 = brain.add_neuron()
        n2 = brain.add_neuron()
        self.assertEqual(2, brain.num_neurons)
        brain.add_synapse(n1, n2, 0.0, 1)
        self.assertEqual(1, brain.num_synapses)

    def test_connection(self):
        brain = Brain()
        n1 = brain.add_neuron()
        n2 = brain.add_neuron()
        brain.add_synapse(n1, n2, 1.0, 1)
        neuron1 = brain.get_neuron(n1)
        neuron2 = brain.get_neuron(n2)
        self.assertEqual(0.0, neuron1.excitement)
        self.assertEqual(0.0, neuron2.excitement)
        neuron1.excite(1.0)
        self.assertEqual(1.0, neuron1.excitement)
        brain.step()
        brain.step()
        self.assertEqual(1.0, neuron2.get_activation(0))

    def test_fully_connected(self):
        arch = [1,2,1]
        brain = fully_connected(arch)
        self.assertEqual(4, brain.num_neurons)
        self.assertEqual(4, brain.num_synapses)

    def test_similar_to_keras(self):
        arch = [1,2,1]
        x = random.random()
        w = random.random()

        # keras
        model = Sequential()
        model.add(Dense(arch[1], activation="linear", input_shape=(arch[0],)))
        model.add(Dense(arch[2], activation="linear"))
        model.layers[0].set_weights([np.full(shape=(1,2), fill_value=w), np.zeros(shape=(2,))])
        model.layers[1].set_weights([np.full(shape=(2,1), fill_value=w), np.zeros(shape=(1,))])
        model.compile("sgd", "mse")
        y_keras = model.predict(np.array([x]))[0][0]

        # brain
        brain = Brain()
        i1 = brain.add_neuron()
        h1 = brain.add_neuron()
        h2 = brain.add_neuron()
        o1 = brain.add_neuron()

        brain.add_synapse(i1, h1, w, 1)
        brain.add_synapse(i1, h2, w, 1)
        brain.add_synapse(h1, o1, w, 1)
        brain.add_synapse(h2, o1, w, 1)

        brain.excite_neuron(i1, x)
        brain.step_n(len(arch))

        y_brain = brain.get_activation([o1])[0]

        self.assertAlmostEqual(y_keras, np.float32(y_brain), places=4)

    def test_similar_to_keras_with_bias(self):
        arch = [1,2,1]
        x = random.random()
        w = random.random()

        # keras
        model = Sequential()
        model.add(Dense(arch[1], activation="relu", input_shape=(arch[0],)))
        model.add(Dense(arch[2], activation="relu"))
        model.layers[0].set_weights([np.full(shape=(1,2), fill_value=w), np.ones(shape=(2,))])
        model.layers[1].set_weights([np.full(shape=(2,1), fill_value=w), np.ones(shape=(1,))])
        model.compile("sgd", "mse")
        y_keras = model.predict(np.array([x]))[0][0]

        # brain
        brain = Brain()
        i1 = brain.add_neuron(activation_fn="linear")
        h1 = brain.add_neuron(activation_fn="relu", bias=1.0)
        h2 = brain.add_neuron(activation_fn="relu", bias=1.0)
        o1 = brain.add_neuron(activation_fn="relu", bias=1.0)

        brain.add_synapse(i1, h1, w, 1)
        brain.add_synapse(i1, h2, w, 1)
        brain.add_synapse(h1, o1, w, 1)
        brain.add_synapse(h2, o1, w, 1)

        brain.excite_neuron(i1, x)
        brain.step_n(len(arch))

        y_brain = brain.get_activation([o1])[0]
        self.assertAlmostEqual(y_keras, np.float32(y_brain), places=4)

    def test_custom_identifier(self):
        brain = Brain()
        n1 = brain.add_neuron(2352354)
        n2 = brain.add_neuron(2422340)
        brain.add_synapse(n1, n2, 1.0, 1)
        neuron1 = brain.get_neuron(n1)
        neuron2 = brain.get_neuron(n2)
        self.assertEqual(0.0, neuron1.excitement)
        self.assertEqual(0.0, neuron2.excitement)
        neuron1.excite(1.0)
        self.assertEqual(1.0, neuron1.excitement)
        brain.step()
        brain.step()
        self.assertEqual(1.0, neuron2.get_activation(0))


if __name__ == '__main__':
    unittest.main()