
import numpy as np

from matrix_factorization.matrix_factorization import MatrixFactorization


if __name__ == '__main__':
    len_x, len_y, len_k = 100, 50, 10
    sample_data = np.zeros((len_x, len_y))
    num_samples = 500
    for _ in range(num_samples):
        val = np.random.randint(1, 11)
        x_idx, y_idx = np.random.randint(0, len_x), np.random.randint(0, len_y)
        while sample_data[x_idx, y_idx] != 0:
            x_idx, y_idx = np.random.randint(0, len_x), np.random.randint(0, len_y)
        sample_data[x_idx, y_idx] = val
    mf = MatrixFactorization(sample_data, len_k, epochs=50)
    mf.train()
