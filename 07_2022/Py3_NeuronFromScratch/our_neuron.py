import numpy as np

from neuron import Neuron

"""
Combine the Neuron class with the sigmoid function
to form a neural network
"""


class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])  # w1 & w2
        bias = 0  # b

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


if __name__ == '__main__':
    network = OurNeuralNetwork()
    x = np.array([2, 3])
    print(network.feedforward(x))  # 0.7216
