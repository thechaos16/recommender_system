
import numpy as np

from multi_armed_bandit.environment import RandomBandit


class MultiArmedBandit:
    def __init__(self, number_of_arms, env=None):
        self.env = RandomBandit(number_of_arms) if not env else env
        self.avg = [0] * number_of_arms
        self.log = []
        for _ in range(number_of_arms):
            self.log.append([0, 0])
        self.avg = np.array(self.avg, dtype=np.float)
        self.regret = 0.0
        self.cnt = 0

    def predict_and_update(self):
        select = self._predict()
        self._update_regret(select)
        self.cnt += 1
        reward = self.env.pull(select)
        self._update(select, reward)

    def _predict(self):
        raise NotImplementedError()

    def _update(self, idx, reward):
        self.log[idx][0] += reward
        self.log[idx][1] += 1
        self.avg[idx] = float(self.log[idx][0]) / float(self.log[idx][1])

    def _update_regret(self, select):
        real_max = np.max(self.env.arms)
        self.regret += real_max - self.env.arms[select]


class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, number_of_arms, env=None, epsilon=0.9):
        super(EpsilonGreedy, self).__init__(number_of_arms, env)
        self.epsilon = epsilon

    def _predict(self):
        val = np.random.random()
        if val > self.epsilon:
            return np.random.choice(range(len(self.avg)))
        return np.argmax(self.avg)

    def _update(self, idx, reward):
        super(EpsilonGreedy, self)._update(idx, reward)


class UCB1(MultiArmedBandit):
    def __init__(self, number_of_arms, env=None):
        super(UCB1, self).__init__(number_of_arms, env)
        self.ucb = np.ones(number_of_arms) * 100

    def _predict(self):
        return np.argmax(self.avg + self.ucb)

    def _update(self, idx, reward):
        super(UCB1, self)._update(idx, reward)
        for idx in range(len(self.env.arms)):
            if self.log[idx][1] != 0:
                self.ucb[idx] = np.sqrt(2 * np.log(self.cnt) / self.log[idx][1])


class ThompsonSampling(MultiArmedBandit):
    def __init__(self, number_of_arms, env=None):
        super(ThompsonSampling, self).__init__(number_of_arms, env)

    def _predict(self):
        max_val = 0
        max_idx = None
        for idx in range(len(self.env.arms)):
            a, b = self.log[idx][0] + 1, self.log[idx][1] - self.log[idx][0] + 1
            beta_sample = np.random.beta(a, b, 1)
            if beta_sample > max_val:
                max_val = beta_sample
                max_idx = idx
        return max_idx


if __name__ == '__main__':
    arms, iters = 10, 500
    env = RandomBandit(arms)
    bandits = [EpsilonGreedy(arms, env, 0.9), UCB1(arms, env), ThompsonSampling(arms, env)]
    # bandit = EpsilonGreedy(arms, 0.9)
    # bandit = UCB1(arms)
    # bandit = ThompsonSampling(arms)
    print(env.arms)
    for bandit in bandits:
        for _ in range(iters):
            bandit.predict_and_update()
        print(bandit.avg)
        print(bandit.log)
        print(bandit.regret / bandit.cnt)
