import gym
from gym import wrappers
import numpy as np
from math import exp
from random import sample

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

load_model = False
save_iter = 20
backup_iter = 500

def create_model():
    model = Sequential()
    model.add(Dense(units=64, input_dim=4, activation='relu'))
    # model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=2, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
    return model

model = create_model()
target_model = create_model()

if load_model:
    model.load_weights("CartPole-v0.keras")

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './tmp/cartpole-experiment-dqn', force=True)
random_env = gym.make('CartPole-v0')

experience_replay = []
gamma_discount = 0.99 # Discounted future reward
epsilon = 1.0 # Probability of random action
epsilon_min = 0.01
episodes = 2000
LAMBDA = 0.001 # Speed of decay
update_target_freq = 1000 # Update weights of target DQN
batch_size = 64
memory_capacity = 100000
steps = 0

def replay(batch_size):
    batch_len = min(batch_size, len(experience_replay))
    # indices = np.random.choice(len(experience_replay), min(batch_size, len(experience_replay)))
    batch = sample(experience_replay, batch_len)

    no_state = np.zeros(env.observation_space.shape[0])

    states = np.array([o[0] for o in batch])
    states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

    pred = model.predict(states)
    pred_ = target_model.predict(states_)

    x = np.zeros((batch_len, env.observation_space.shape[0]))
    y = np.zeros((batch_len, env.action_space.n))

    for i in range(batch_len):
        o = batch[i]
        old_observation = o[0]
        action = o[1]
        reward = o[2]
        observation = o[3]

        target = pred[i]

        if observation is None:
            target[action] = reward
        else:
            target[action] = reward + gamma_discount * np.amax(pred_[i])

        x[i] = old_observation
        y[i] = target

    model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0)

# Fill experience replay memory with random actions initially
while len(experience_replay) < memory_capacity:
    observation = random_env.reset()

    R = 0

    while True:
        action = np.random.randint(random_env.action_space.n)

        old_observation = observation
        observation, reward, done, info = random_env.step(action)

        if done:
            observation = None

        if len(experience_replay) % 1000 == 0:
            print('Experience Replay Size: ', len(experience_replay))

        if len(experience_replay) > memory_capacity:
            experience_replay.pop(0)
        experience_replay.append([old_observation, action, reward, observation])

        R += reward

        if done:
            print("Total reward:", R)
            break


for episode in range(episodes):
    observation = env.reset()

    R = 0
    i = 0

    while True:
        action = model.predict(np.reshape(observation, (1, env.observation_space.shape[0])))
        action = np.argmax(action[0])

        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)

        old_observation = observation
        observation, reward, done, info = env.step(action)

        if done:
            observation = None

        if len(experience_replay) > memory_capacity:
            experience_replay.pop(0)
        experience_replay.append([old_observation, action, reward, observation])

        if steps % 1000 == 0:
            target_model.set_weights(model.get_weights())

        replay(batch_size)

        R += reward
        steps += 1
        i += 1
        epsilon = epsilon_min + (epsilon - epsilon_min) * exp(-LAMBDA * steps)

        if done:
            print('Episode {} finished in {} iterations.'.format(episode, i))
            print("Total reward:", R)
            break

    if episode % save_iter == 0:
        model.save_weights('CartPole-v0.keras')
    if episode % backup_iter == 0:
        model.save_weights("CartPole_backup" + str(episode) + "-v0.keras")


