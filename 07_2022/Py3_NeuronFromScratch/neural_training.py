import numpy as np
from neuron import Neuron, sigmoid


def deriv_sigmoid(x):
    # Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # Calculate the mean squared error: mean(square(y - y'))
    return np.mean((y_true - y_pred) ** 2)


class TrainableNeuralNetwork:

    # Initialize the neural network
    # 2 inputs (x1, x2)
    # 1 output (y)
    # 1 hidden layer with 2 neurons (h1 & h2)
    def __init__(self):
        # weights
        # w1 & w2 = x1 & x2 -> h1
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()

        # w3 & w4 = x1 & x2 -> h1
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()

        # w5 = h1 -> y
        self.w5 = np.random.normal()

        # w6 = h2 -> y
        self.w6 = np.random.normal()

        # biases
        self.b1 = np.random.normal()  # h1
        self.b2 = np.random.normal()  # h2
        self.b3 = np.random.normal()  # y

        self.h1 = Neuron(np.array([self.w1, self.w2]), self.b1)
        self.h2 = Neuron(np.array([self.w3, self.w4]), self.b2)
        self.y = Neuron(np.array([self.w5, self.w6]), self.b3)

    def feedforward(self, x):
        o_h1 = self.h1.feedforward(x)
        o_h2 = self.h2.feedforward(x)
        y = self.y.feedforward(np.array([o_h1, o_h2]))
        return y

    def train(self, data, all_y_trues):
        # Implement the training loop here
        # 1. Feedforward
        # 2. Backpropagation
        # 3. Update the weights

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.h1.sum(x)
                o_h1 = sigmoid(sum_h1)

                sum_h2 = self.h2.sum(x)
                o_h2 = sigmoid(sum_h2)

                sum_y = self.y.sum(np.array([o_h1, o_h2]))
                o_y = sigmoid(sum_y)
                y_pred = o_y

                # Calculate partial derivation
                # d_a_d_b = "partial a / partial b"
                d_loss_d_y_pred = -2 * (y_true - y_pred)

                # o1
                d_y_pred_d_w5 = o_h1 * deriv_sigmoid(sum_y)
                d_y_pred_d_w6 = o_h2 * deriv_sigmoid(sum_y)
                d_y_pred_d_b3 = deriv_sigmoid(sum_y)

                d_y_pred_d_h1 = self.w5 * deriv_sigmoid(sum_y)
                d_y_pred_d_h2 = self.w6 * deriv_sigmoid(sum_y)

                # h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                # h1
                self.w1 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h1 * d_h1_d_b1
                self.h1 = Neuron(np.array([self.w1, self.w2]), self.b1)

                # h2
                self.w3 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_h2 * d_h2_d_b2
                self.h2 = Neuron(np.array([self.w3, self.w4]), self.b2)

                # y
                self.w5 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_w5
                self.w6 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_w6
                self.b3 -= learn_rate * d_loss_d_y_pred * d_y_pred_d_b3
                self.y = Neuron(np.array([self.w5, self.w6]), self.b3)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch: %d, Loss: %.3f" % (epoch, loss))

    def predict(self, x):
        result = self.feedforward(x)
        print("%.0f" % result)


if __name__ == '__main__':
    # Define dataset

    # [height - 135, weight - 66]
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])

    # [gender], 0 = male, 1 = female
    all_y_trues = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    network = TrainableNeuralNetwork()
    network.train(data, all_y_trues)
    network.predict(np.array([-2, -1]))
