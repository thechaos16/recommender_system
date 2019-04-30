
import random


class RandomBandit:
    def __init__(self, number_of_arms):
        self.arms = list()
        self.initialize(number_of_arms)

    def initialize(self, number):
        for _ in range(number):
            self.arms.append(random.random())

    def pull(self, idx):
        rand = random.random()
        return 1 if self.arms[idx] > rand else 0
