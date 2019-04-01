
import numpy as np


class MatrixFactorization:
    def __init__(self, data, hidden_dim, learning_rate=0.1, beta=0.01, iterations=100, seed=42, bias=False):
        self.data = data
        self.hidden_dim = hidden_dim
        self.input_dim, self.output_dim = data.shape
        self.lr = learning_rate
        self.beta = beta
        self.iterations = iterations
        self.bias = bias
        self.random_seed = seed
        np.random.seed(self.random_seed)
        self.p_mat, self.q_mat = self.initialize()
        self.b, self.b_x, self.b_y = self.initialize_bias()

    def initialize(self):
        return np.random.random((self.input_dim, self.hidden_dim)), \
               np.random.random((self.hidden_dim, self.output_dim))

    def initialize_bias(self):
        if self.bias:
            return 0, np.zeros(self.input_dim), np.zeros(self.output_dim)
        return 0, np.zeros(self.input_dim), np.zeros(self.output_dim)

    def train(self):
        pass

    def get_mse(self):
        pass

    def get_gradient(self):
        pass

    def reconstruct(self):
        res = np.matmul(self.p_mat, self.q_mat) + self.b +
        return res
