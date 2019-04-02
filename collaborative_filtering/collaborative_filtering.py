
import numpy as np

from matrix_factorization.matrix_factorization import MatrixFactorization


class CollaborativeFiltering:
    def __init__(self, data, algorithm='matrix_factorization', similarity='cosine'):
        self.data = data
        self.alg, self.sim = self.initialize(algorithm, similarity)

    def initialize(self, alg, sim):
        if alg == 'matrix_factorization':
            method = MatrixFactorization(self.data, 20)
        else:
            raise NotImplementedError()
        if sim == 'cosine':
            sim_method = get_cosine_similarity
        else:
            raise NotImplementedError()
        return method, sim_method

    def fit(self):
        self.alg.train()

    def predict(self, new_user):
        full_mat = self.alg.reconstruct()
        max_sim = 0
        matched_user = None
        for row in full_mat:
            cur_sim = self.sim(row, new_user)
            if cur_sim > max_sim:
                matched_user = row
                max_sim = cur_sim
        max_idx = np.argmax(matched_user)
        return matched_user, max_idx


def get_cosine_similarity(row1, row2):
    assert len(row1) == len(row2)
    row1, row2 = np.array(row1), np.array(row2)
    cosine = np.dot(row1, row2) / np.sqrt(np.sum(row1 ** 2) * np.sum(row2 ** 2))
    return cosine
