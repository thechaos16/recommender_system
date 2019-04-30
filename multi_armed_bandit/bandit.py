
import numpy as np

from multi_armed_bandit.environment import RandomBandit


class MultiArmedBandit:
    def __init__(self, number_of_arms):
        self.env = RandomBandit(number_of_arms)
        self.avg = []
        self.log = []
        for _ in range(number_of_arms):
            self.avg.append([0])
            self.log.append([0, 0])
        self.avg = np.array(self.avg, dtype=np.float)

    def predict(self):
        raise NotImplementedError()

    def update(self, idx, reward):
        self.log[idx][0] += reward
        self.log[idx][1] += 1


class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, number_of_arms, epsilon):
        super(EpsilonGreedy, self).__init__(number_of_arms)
        self.epsilon = epsilon

    def predict(self):
        val = np.random.random()
        if val > self.epsilon:
            return np.random.choice(range(len(self.avg)))
        return np.argmax(self.avg)

    def update(self, idx, reward):
        super(EpsilonGreedy, self).update(idx, reward)
        self.avg[idx] = float(self.log[idx][0]) / float(self.log[idx][1])


if __name__ == '__main__':
    bandit = EpsilonGreedy(10, 0.9)
    print(bandit.env.arms)
    for _ in range(1000):
        one = bandit.predict()
        reward = bandit.env.pull(one)
        bandit.update(one, reward)
    print('Updated!!')
    print(bandit.avg)
    print(bandit.log)

    rewards, cnt = 0, 0
    res = {idx: 0 for idx in range(10)}
    for _ in range(100):
        one = bandit.predict()
        res[one] += 1
        reward = bandit.env.pull(one)
        rewards += reward
        cnt += 1
    print(res)
    print(rewards, cnt)
