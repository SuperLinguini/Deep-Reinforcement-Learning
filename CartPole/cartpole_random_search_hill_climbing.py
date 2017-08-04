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


def random_search(env):
    env = wrappers.Monitor(env, './tmp/cartpole-experiment-random-search', force=True)

    best_weights = None
    best_reward = 0

    for _ in range(10000):
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(env, weights)
        if reward > best_reward:
            best_reward = reward
            best_weights = weights
        if reward > 200:
            break
    return best_reward, best_weights


def hill_climbing(env):
    env = wrappers.Monitor(env, './tmp/cartpole-experiment-hill-climbing', force=True)

    exploration_noise = 0.1
    weights = np.random.rand(4) * 2 - 1
    best_reward = 0

    for _ in range(10000):
        new_weights = weights + (np.random.rand(4) * 2 - 1) * exploration_noise
        reward = run_episode(env, new_weights)

        if reward > best_reward:
            best_reward = reward
            weights = new_weights

            if reward > 200:
                break

    return best_reward, weights


def main():
    env = gym.make('CartPole-v0')

    best_reward, best_weights = random_search(env)

    print('Best weights - Random Search: ', best_weights)
    print('Highest reward - Random Search: ', best_reward)

    best_reward, best_weights = hill_climbing(env)

    print('Best weights - Hill Climbing: ', best_weights)
    print('Highest reward - Hill Climbing: ', best_reward)


if __name__ == '__main__':
    main()
