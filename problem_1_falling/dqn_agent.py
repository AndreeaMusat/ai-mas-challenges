# Andreea Musat, October 2018

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from typing import Dict

from dqn import DQN
from memory import Experience
from frame_processor import stack_frames

np.set_printoptions(threshold=np.nan)


class DQNAgent:
    def __init__(self, max_action: int, training=False):
        self.max_action = max_action
        self.sess = None

    def reset_state(self, observation: np.ndarray):
        self.stacked_frames, self.state = stack_frames(deque([], maxlen=self.stack_size), observation,
                                                       self.frame_size, self.stack_size)

    def do_setup(self, args: Dict, observation: np.ndarray, session: tf.Session):
        self.frame_size = args.frame_size
        self.stack_size = args.stack_size

        self.mem = Experience(capacity=args.mem_capacity)
        self.dqn = DQN((*args.frame_size, args.stack_size),
                       self.max_action, args.learning_rate, "ex")
        self.stacked_frames, self.state = stack_frames(
            deque([], maxlen=self.stack_size), observation, args.frame_size, args.stack_size)

        self.sess = session
        self.sess.run(tf.global_variables_initializer())

    def remember(self, observation: np.ndarray, action: int, reward: float):
        """
        Add current observation to the stack of frames and create a memory
        entry corresponding to this tuple. Also update the internal state of
        the agent.
        """
        self.stacked_frames, next_state = stack_frames(self.stacked_frames, observation,
                                                       self.frame_size, self.stack_size)
        self.mem.add((self.state, action, reward, next_state))
        self.state = next_state

    def get_random_action(self):
        return np.random.randint(self.max_action)

    # TODO: move explore_prob_* and decay rate inside the agent
    # This should be used when the agent is still in training.
    def predict_action(self, explore_prob_begin: float, explore_prob_min: float, decay_rate: float, decay_step: int):
        explore_prob_curr = explore_prob_min + \
            (explore_prob_begin - explore_prob_min) * \
            np.exp(-decay_rate * decay_step)
        if np.random.rand() < explore_prob_curr:
            action = self.get_random_action()
        else:
            Qs = self.sess.run(self.dqn.output, feed_dict={
                               self.dqn.inputs_: self.state.reshape(1, *self.state.shape)})
            action = int(np.argmax(Qs))
            print('action: %d' % action)

        return action, explore_prob_curr

    def act(self, observation: np.ndarray):
        """
        :param observation: numpy array of shape (width, height, 3) *defined in config file
        :return: int between 0 and max_action
        This method should be called when the agent is already trained.
        """

        if self.sess is None:
            # Used some hardcoded parameters here, sorryyy.
            session = tf.Session()
            self.dqn = DQN((*(84, 84), 4), self.max_action, 0.002, "ex")
            saver = tf.train.Saver()
            saver.restore(session, './models/second_model.ckpt')
            self.sess = session
            self.cnt = 0
            self.stacked_frames, self.state = stack_frames(
                deque([], maxlen=4), observation, (84, 84), 4)

        # Used to visualize the game when testing the model.
        cv2.imwrite('game_' + str(self.cnt) + '.png', observation)
        self.cnt += 1

        self.stacked_frames, self.state = stack_frames(self.stacked_frames, observation,
                                                       (84, 84), 4)

        Qs = self.sess.run(self.dqn.output, feed_dict={
                           self.dqn.inputs_: self.state.reshape(1, *self.state.shape)})
        return int(np.argmax(Qs))
