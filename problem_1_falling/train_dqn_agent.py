# Andreea Musat, October 2018

import argparse
import cv2
import numpy as np
import tensorflow as tf

from collections import deque
from typing import Tuple

from dqn import DQN
from dqn_agent import DQNAgent
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
from utils import read_cfg


batch_size = 32
num_exploration_steps = 1000
explore_prob_begin = 1.0   # Just explore in the beginning
explore_prob_min = 0.01    # Make it stick to its policy later.
decay_rate = 0.001


def train_dqn_model(args):
    action_size = max(ACTIONS.keys()) + 1
    env = FallingObjects(read_cfg(args.config_file))
    obs = env.reset()

    tf.reset_default_graph()

    with tf.Session() as sess:
        # Create and initialize the agent.
        agent = DQNAgent(action_size, training=True)
        agent.do_setup(args, obs, sess)

        # Tensorboard setup.
        writer = tf.summary.FileWriter("./logs")
        saver = tf.train.Saver()
        tf.summary.scalar("Loss", agent.dqn.loss)
        write_op = tf.summary.merge_all()

        # Now start learning.
        obs = env.reset()
        all_rewards = []

        # We first play a bit in order to explore the environment
        # and populate the experience buffer.
        for i in range(num_exploration_steps):
            action = agent.get_random_action()
            obs, reward, _, _ = env.step(action)
            all_rewards.append(reward)
            total_reward = sum(all_rewards[-args.stack_size:])
            # total_reward = reward
            agent.remember(obs, action, total_reward)

        all_rewards = []
        for step in range(args.num_train_steps):
            # Predict an action using an e-greedy policy, where the
            # probability of exploration is decaying in time.
            action, explore_prob = agent.predict_action(explore_prob_begin,
                                                        explore_prob_min, decay_rate, step)

            # Apply the action and get the observation and reward from
            # the environment.
            obs, reward, _, _ = env.step(action)
            all_rewards.append(reward)

            # Save the current observation to see how the agent behaves.
            cv2.imwrite(str(step) + '.png', obs)

            # And make this part of the agent's experience.
            total_reward = sum(all_rewards[-args.stack_size:])
            agent.remember(obs, action, total_reward)
            print('Step %7d, total reward = %2d' % (step, total_reward))

            # Get a mini-batch from memory and train the net.
            mini_batch = agent.mem.sample(batch_size)
            states, actions, rewards, next_states = (
                list(elem) for elem in zip(*mini_batch))

            # Compute one-host encodings for the actions.
            actions_one_hot = np.zeros((len(actions), action_size))
            actions_one_hot[np.arange(len(actions)), actions] = 1

            target_Qs = []

            # Q values for the next states using.
            next_Qs = agent.sess.run(agent.dqn.output,
                                     feed_dict={agent.dqn.inputs_: next_states})

            # Q target should be reward + gamma * maxQ(s', a')
            target_Qs = np.array([rewards[i] + args.discount_factor * np.max(next_Qs[i])
                                  for i in range(batch_size)])

            loss, _ = agent.sess.run([agent.dqn.loss, agent.dqn.optimizer],
                                     feed_dict={agent.dqn.inputs_: states,
                                                agent.dqn.target_Q: target_Qs,
                                                agent.dqn.actions_: actions_one_hot})

            summary = sess.run(write_op,
                               feed_dict={agent.dqn.inputs_: states,
                                          agent.dqn.target_Q: target_Qs,
                                          agent.dqn.actions_: actions_one_hot})

            writer.add_summary(summary, step)
            writer.flush()

            # Save the model every 10 steps.
            if step % 10 == 0:
                saver.save(sess, './models/' + args.model_name + '.ckpt')


def process_args(args):
    args.frame_size = tuple([int(x) for x in args.frame_size])
    args.stack_size = int(args.stack_size)
    args.learning_rate = float(args.learning_rate)
    args.discount_factor = float(args.discount_factor)
    args.mem_capacity = int(args.mem_capacity)
    args.num_train_steps = int(args.num_train_steps)

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-size', type=int, nargs=2, dest='frame_size')
    parser.add_argument('--stack-size', type=int, dest='stack_size')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate')
    parser.add_argument('--discount-factor', type=float,
                        dest='discount_factor')
    parser.add_argument('--mem-capacity', type=int, dest='mem_capacity')
    parser.add_argument('--model-name', type=str, dest='model_name')
    parser.add_argument('--config-file', type=str, dest='config_file')
    parser.add_argument('--num-train-steps', type=int, dest='num_train_steps')

    train_dqn_model(process_args(parser.parse_args()))
