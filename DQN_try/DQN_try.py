import __future__
# import gym
# from gym.wrappers import Monitor

import gym
import gym_icegame 

import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple

# env from: https://github.com/kvzhao/icegame
env = gym.make('IceGameEnv-v0')
VALID_ACTIONS = [0, 1, 2, 3, 4, 5, 6]
# VALID_ACTIONS = [0, 1, 2, 3, 4, 5]  ## AUTO 6



class StateProcessor():
    def __init__(self):
        # # Build the Tensorflow graph
        # with tf.variable_scope("state_processor"):
        #     self.input_state = tf.placeholder(shape=[32, 32, 1], dtype=tf.int32)    #uint8
        #     self.output = tf.image.rgb_to_grayscale(self.input_state)
        #     # self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
        #     # self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #     self.output = tf.squeeze(self.output)
        #     # self.output = tf.squeeze(self.input_state)
        self.test = 1

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return state    #sess.run(self.output, { self.input_state: state })



class Estimator():

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        # Placeholders for our input
        self.X_pl = tf.placeholder(shape=[None, 32, 32, 4, 4], dtype=tf.int32, name="X")   # uint8
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # X = tf.to_float(self.X_pl) / 255.0
        X = tf.to_float(self.X_pl)   
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(X, 32, 4, 1, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 32, 3, 1, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 32, 2, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)    # conv3
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)

        ## for predict
        self.predictions = tf.contrib.layers.fully_connected(fc2, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


# For Testing....

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    # observation = env.new_random_game()

    observation_p = sp.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)
    
    # Test Prediction
    print('predict: ', e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print('update: ', e.update(sess, observations, a, y))


class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.  
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))   # env.spec.id

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    max_reward = 0

    # for 2.7
    q_estimator=q_estimator
    target_estimator=target_estimator
    state_processor=state_processor
    experiment_dir=experiment_dir
    num_episodes=100000
    replay_memory_size=800000
    replay_memory_init_size=50000
    update_target_estimator_every=10000
    epsilon_start=1.0
    epsilon_end=0.1
    epsilon_decay_steps=500000
    discount_factor=0.99
    batch_size=32
    try_times=1024*4

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []
    
    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    # # Record videos
    # # Add env Monitor wrapper
    # env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % record_video_every == 0, resume=True)

    ## GO from Here
    print("Game start")

    # ## accept record initialize
    # env.sim.init_model()

    max_reward = -299792458
    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            # print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss), end="")
            
            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state, reward, done, action_dict = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # action_list.append(action)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            ## ORI DQN
            # # Calculate q values and targets
            # q_values_next = target_estimator.predict(sess, next_states_batch)
            # targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

            # Calculate q values and targets
            # This is where Double Q-Learning comes in!
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            this_time_reward = stats.episode_rewards[i_episode]
            if this_time_reward > max_reward:
                max_reward = this_time_reward

            # if done or t >= try_times:
                # if reward > 2:
                #     env.render()
            if done:
                print("\rStep {} ({}) @ Episode {}/{}, loss: {}, reward: {}, max_reward: {}".format(t, total_t, i_episode + 1, num_episodes, loss, this_time_reward, max_reward))
                print('\raction_dict: {}'.format(action_dict))
                sys.stdout.flush()
                break

            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        # episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
        # episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_writer.flush()
        
        # yield total_t, plotting.EpisodeStats(
        #     episode_lengths=stats.episode_lengths[:i_episode+1],
        #     episode_rewards=stats.episode_rewards[:i_episode+1])




    # for t, this_reward in 

    #     if this_reward > max_reward:
    #         max_reward = this_reward
    #     print('Episode Reward: ', this_reward, ', Max Reward: ', max_reward, '\n')


    # for t, stats in deep_q_learning(sess,
    #                                 env,
    #                                 q_estimator=q_estimator,
    #                                 target_estimator=target_estimator,
    #                                 state_processor=state_processor,
    #                                 experiment_dir=experiment_dir,
    #                                 num_episodes=50000,
    #                                 replay_memory_size=800000,
    #                                 replay_memory_init_size=50000,
    #                                 update_target_estimator_every=10000,
    #                                 epsilon_start=1.0,
    #                                 epsilon_end=0.1,
    #                                 epsilon_decay_steps=500000,
    #                                 discount_factor=0.99,
    #                                 batch_size=32):

    #     this_reward = stats.episode_rewards[-1]

    #     if this_reward > max_reward:
    #         max_reward = this_reward

    #     # print("\nEpisode Reward: {}".format(this_reward))
    #     print('Episode Reward: ', this_reward, ', Max Reward: ', max_reward, '\n')




    


















