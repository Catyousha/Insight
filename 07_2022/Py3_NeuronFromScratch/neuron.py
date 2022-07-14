import numpy as np


def sigmoid(x):
    # f(x) = 1 / (1 + e^(-x))
    # menormalisasikan nilai tak terbatas jadi 0-1
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sum(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return total

    def feedforward(self, inputs):
        total = self.sum(inputs)
        return sigmoid(total)


# Konsep dasar neuron
# f(x1, x2) = w1x1 + w2x2 + b
if __name__ == '__main__':
    w = np.array([0, 1])  # w1 & w2
    b = 4  # b
    neuron = Neuron(w, b)

    x = np.array([2, 3])  # x1 & x2
    print(neuron.feedforward(x))
