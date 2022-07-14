import numpy as np


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


if __name__ == '__main__':
    y_real = np.array([0, 0, 1, 1])
    y_pred = np.array([0.5, 0.5, 0.9, 0.9])
    print(mse_loss(y_real, y_pred))  # 0.13, the lower, the better
