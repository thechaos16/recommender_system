
import numpy as np


class MatrixFactorization:
    def __init__(self, data, hidden_dim, learning_rate=0.01, beta=0.1, epochs=100, seed=42, bias=False):
        self.data = data
        self.hidden_dim = hidden_dim
        self.input_dim, self.output_dim = data.shape
        self.lr = learning_rate
        self.beta = beta
        self.epochs = epochs
        self.bias = bias
        self.random_seed = seed
        np.random.seed(self.random_seed)
        self.p_mat, self.q_mat = self.initialize()
        self.b, self.b_x, self.b_y = self.initialize_bias()

    def initialize(self):
        return np.random.random((self.input_dim, self.hidden_dim)), \
               np.random.random((self.hidden_dim, self.output_dim))

    def initialize_bias(self):
        return 0, np.zeros(self.input_dim), np.zeros(self.output_dim)

    def train(self):
        for epoch in range(self.epochs):
            predicted = self.reconstruct()
            mse = self.get_mse(predicted)
            print('epoch: {}, MSE: {}'.format(epoch, mse))
            self.update(predicted)
        print('Final MSE: {}'.format(self.get_mse(self.reconstruct())))

    def get_mse(self, predicted):
        err, cnt = 0, 0
        for row in range(self.input_dim):
            for col in range(self.output_dim):
                if self.data[row, col] == 0:
                    continue
                err += (self.data[row, col] - predicted[row, col]) ** 2
                cnt += 1
        err /= cnt
        return np.sqrt(err)

    def update(self, predicted):
        for row in range(self.input_dim):
            for col in range(self.output_dim):
                if self.data[row, col] == 0:
                    continue
                err = self.data[row, col] - predicted[row, col]
                self.b_x[row] += self.lr * (err - self.beta * self.b_x[row])
                self.b_y[col] += self.lr * (err - self.beta * self.b_y[col])
                self.p_mat[row, :] += self.lr * (err * self.q_mat[:, col] - self.beta * self.p_mat[row, :])
                self.q_mat[:, col] += self.lr * (err * self.p_mat[row, :] - self.beta * self.q_mat[:, col])

    def reconstruct(self):
        res = np.matmul(self.p_mat, self.q_mat) + self.b + np.tile(self.b_x, (self.output_dim, 1)).T \
              + np.tile(self.b_y, (self.input_dim, 1))
        return res
