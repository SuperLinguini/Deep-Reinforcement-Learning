import gym
from gym import wrappers
import numpy as np
from math import exp

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, sgd, RMSprop

load_model = 1
save_iter = 5
backup_iter = 500
memory_clear = 100

model = Sequential()
model.add(Dense(units=128, input_dim=4, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2))
model.compile(loss='mse', optimizer=RMSprop(lr=0.0001))

if load_model == 1:
    model.load_weights("cartpole-v0.keras")

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './tmp/cartpole-experiment-dqn', force=True)

experience_replay = []
gamma_discount = 0.99 # Discounted future reward
epsilon = 1.0 # Probability of random action
epsilon_min = 0.01
episodes = 10000
LAMBDA = 0.001 # Speed of decay
steps = 0

def replay(batch_size):
    print('Experience Replay Size: ', len(experience_replay))
    indices = np.random.choice(len(experience_replay), min(batch_size, len(experience_replay)))

    for index in indices:
        old_observation, action, reward, observation = experience_replay[index]
        target = reward
        if index != len(experience_replay) - 1:
            target = reward + gamma_discount * np.max(model.predict(observation)[0])
        print("Target:", target)

        target_f = model.predict(old_observation)
        target_f[0][action] = target

    model.fit(old_observation, target_f, epochs=1, verbose=0)

for episode in range(episodes):
    observation = env.reset()
    observation = np.reshape(observation, (1,4))

    for i in range(5000):
        print(epsilon)
        env.render()

        action = model.predict(observation)
        print(action)
        action = np.argmax(action[0])

        if np.random.random() < epsilon:
            action = np.random.randint(2)

        old_observation = observation
        observation, reward, done, info = env.step(action)
        observation = np.reshape(observation, (1,4))
        print(observation)

        experience_replay.append([old_observation, action, reward, observation])

        replay(64)

        steps += 1
        epsilon = epsilon_min + (epsilon - epsilon_min) * exp(-LAMBDA * steps)

        if done:
            print('Episode {} finished in {} iterations.'.format(episode, i))
            break

    if episode % save_iter == 0:
        model.save_weights('CartPole-v0.keras')
    if episode % backup_iter == 0:
        model.save_weights("CartPole_backup" + str(episode) + "-v0.keras")
    if episode % memory_clear == 0:
        experience_replay = []


