import tensorflow as tf
import numpy as np

from typing import Tuple
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import xavier_initializer_conv2d

class DQN:
    def __init__(self, state_size: Tuple[int, int, int], action_size: int,
                 learning_rate: float, name: str):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        self.inputs_ = tf.placeholder(
            tf.float32, [None, *state_size], name="inputs")
        self.actions_ = tf.placeholder(
            tf.float32, [None, self.action_size], name="actions_")

        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")

        self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32,
                                      kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                                      kernel_initializer=xavier_initializer_conv2d(),
                                      name="conv1")

        self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                             training=True,
                                                             epsilon=1e-5,
                                                             name='batch_norm1')

        self.conv1_out = tf.nn.relu(self.conv1_batchnorm, name="conv1_out")
        self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                      filters=64,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding="VALID",
                                      kernel_initializer=xavier_initializer_conv2d(),
                                      name="conv2")

        self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                             training=True,
                                                             epsilon=1e-5,
                                                             name='batch_norm2')

        self.conv2_out = tf.nn.relu(self.conv2_batchnorm, name="conv2_out")
        self.flatten = tf.layers.flatten(self.conv2_out)
        self.fc = tf.layers.dense(inputs=self.flatten,
                                  units=512,
                                  activation=tf.nn.elu,
                                  kernel_initializer=xavier_initializer(),
                                  name="fc1")

        self.output = tf.layers.dense(inputs=self.fc,
                                      kernel_initializer=xavier_initializer(),
                                      units=self.action_size,
                                      activation=tf.nn.softmax)

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
