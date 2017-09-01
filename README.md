# Deep Reinforcement Learning

Personal project to apply deep neural networks to reinforcement learning and create video game AI with Asynchronous Advantage Actor-Critic (A3C) model and Deep Q Networks for OpenAI Gym reinforcement learning environments with TensorFlow/Keras. I only ran it on the CartPole problem but would love to try on more complex domains like Atari with convolutional neural nets when I get a stronger computer.

## Installation

You must have `gym`, `tensorflow`, `numpy`, and `keras`. Had some issues running on Windows, especially with gym wrappers but ran well on *nix (I used Ubuntu).

## Usage

Simply run `python [file-name].py`!

## References

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.