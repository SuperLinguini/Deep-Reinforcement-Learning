import gym
import numpy as np
from gym import wrappers


def run_episode(env, weights):
    observation = env.reset()
    total_reward = 0

    for _ in range(200):
        action = 0 if np.matmul(weights, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


def main():
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './tmp/cartpole-experiment-random-search', force=True)

    best_weights = None
    best_reward = 0

    for _ in range(10000):
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(env, weights)
        if reward > best_reward:
            best_reward = reward
            best_weights = weights
        if reward == 300:
            break

    print('Best weights: ', best_weights)
    print('Highest reward: ', best_reward)


if __name__ == '__main__':
    main()
