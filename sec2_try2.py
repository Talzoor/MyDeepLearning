import random
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from datetime import datetime
from log import Log
import gym


# custom Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.model = None
        self.TB_graph = tf.compat.v1.Graph()
        with self.TB_graph.as_default():
            self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=5000)
            self.writer.set_as_default()
            self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        self.TB_sess = tf.compat.v1.InteractiveSession(graph=self.TB_graph)
        self.TB_sess.run(self.writer.init())

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '\\train'

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # added for performance?
    def on_train_batch_end(self, _, __):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            self.TB_sess.run(self.all_summary_ops)
            if self.model is not None:
                name = f'{name}_{self.model.name}'
            self.TB_sess.run(tf.summary.scalar(name, value, step=index))
        self.model = None


class ExperienceReplay:
    """
    Replay Memory
    Stores and retrieves experiences
    """

    def __init__(self, _mem_size, _batch_size):
        self.experiences = deque(maxlen=_mem_size)
        self.batch_size = _batch_size

    def store(self, state, action, reward, next_state, done):
        """
        Records a single step (state transition) of gameplay experience.

        :param state: the current game state
        :param next_state: the game state after taking action
        :param reward: the reward taking action at the current state brings
        :param action: the action taken at the current state
        :param done: a boolean indicating if the game is finished after
        taking the action
        :return: None
        """
        self.experiences.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """
        Samples a batch of gameplay experiences for training.
        :return: a list of gameplay experiences
        """
        if len(self.experiences) < self.batch_size:
            return 0
        batch_min_size = min(self.batch_size, len(self.experiences))
        sampled_batch = random.sample(self.experiences, batch_min_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for data in sampled_batch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            done_batch.append(data[4])
        st_np = np.array(state_batch)
        ac_np = np.array(action_batch)
        rd_np = np.array(reward_batch)
        nx_st_np = np.array(next_state_batch)
        dn_np = np.array(done_batch)
        return st_np, ac_np, rd_np, nx_st_np, dn_np


def custom_loss(y_true, y_pred):
    # make sure that loss is calculated as described (Algo. line #15)
    # calculating squared difference between target and predicted values
    loss = K.square(y_pred - y_true)  # (batch_size, 2)
    return loss


class NeuralNetwork:
    """
    DQN NeuralNetwork
    DQN agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """

    def __init__(self, _env_st_size, _env_ac_size, _lr, _layers, _optimizer, _initializer):
        self.q_net = self._build_dqn_model(_env_st_size, _env_ac_size,
                                           _lr, _layers, _optimizer, _initializer, "Q_Net")
        self.target_q_net = self._build_dqn_model(_env_st_size, _env_ac_size,
                                                  _lr, _layers, _optimizer, _initializer, "Q_Target_Net")

    @staticmethod
    def _build_dqn_model(_env_st_size, _env_ac_size,
                         _learning_rate, _layers, _optimizer, _initializer,
                         _name=None
                         ):
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.
        :return: Q network
        """

        state_size = _env_st_size
        action_size = _env_ac_size

        n_network = Sequential(name=_name if (_name is not None) else None)
        # build with no. of layers given
        n_network.add(Dense(_layers[0],
                            input_dim=state_size,
                            activation='relu',
                            kernel_initializer=_initializer))
        for l in range(1, len(_layers)):
            n_network.add(Dense(_layers[l],
                                activation='relu',
                                kernel_initializer=_initializer))

        #output layer with fixed (action_size) output size.
        n_network.add(Dense(action_size, activation='linear', kernel_initializer=_initializer))
        # n_network.add(Dense(action_size, activation='sigmoid', kernel_initializer=_initializer))

        if _optimizer is "RMSprop":
            # n_network.compile(loss='mse', optimizer=RMSprop(lr=_learning_rate), metrics=['mae'])
            n_network.compile(loss=custom_loss, optimizer=RMSprop(lr=_learning_rate), metrics=['mae'])
        elif _optimizer is "SGD":
            # n_network.compile(loss='mse', optimizer=SGD(lr=_learning_rate))
            n_network.compile(loss=custom_loss, optimizer=SGD(lr=_learning_rate), metrics=['mae'])
        elif _optimizer is "Adam":
            # n_network.compile(loss='mse', optimizer=Adam(lr=_learning_rate))
            n_network.compile(loss=custom_loss, optimizer=Adam(lr=_learning_rate), metrics=['mae'])
            # n_network.compile(loss=custom_loss, optimizer=Adam(lr=_learning_rate))
        elif _optimizer is "Adadelta":
            n_network.compile(loss=custom_loss, optimizer=Adadelta(lr=_learning_rate), metrics=['mae'])
            # n_network.compile(loss='mse', optimizer=Adadelta(lr=_learning_rate))

        return n_network

    def greedy_e_policy(self, _state, _eps):
        """
        Similar to policy but with some randomness to encourage exploration.

        :param state: the game state
        :return: action
        """
        if np.random.random() <= _eps:
            action = np.random.randint(2)
        else:
            action = self.policy(_state)
        return action

    def policy(self, _state):
        """
        Takes a state from the game environment and returns an action that
        has the highest Q value and should be taken as the next step.

        :param _state: the current game environment state
        :return: an action
        """
        # state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)

        # state_np = np.array(_state).reshape(-1, *_state.shape)
        state_np = _state.reshape(1, 4)

        # action_q = self.q_net(state)
        # action_q = self.q_net(state_input)

        # action_q = self.q_net.predict(state_np)
        action_q = self.q_net.predict(state_np).reshape(2, )

        # act_q_np = action_q.numpy()
        action = np.argmax(action_q)
        # print("action:{}".format(action))
        return action

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.

        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, _batch, _gamma):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param _batch: a batch of gameplay experiences
        :return: training loss
        """
        if _batch is 0:
            return 0.0
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = _batch
        # current_states = np.array([data[0] for data in sample_batch])
        # next_state_batch = np.array(next_state_batch).reshape(-1, *next_state_batch.shape)
        current_q = self.q_net.predict(state_batch)
        # current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)

        # next_q = self.target_q_net(next_state_batch).numpy()
        next_q = self.target_q_net.predict(next_state_batch)

        max_next_q = np.amax(next_q, axis=1)
        # max_next_q = np.amax(next_q)
        batch_size = state_batch.shape[0]
        for i in range(batch_size):   # proccess the minibatch
            # target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val = reward_batch[i] + _gamma * max_next_q[i]     # algo. line 13
            else:
                target_q_val = reward_batch[i]                              # algo. line 14

            target_q[i][action_batch[i]] = target_q_val                     # y_j = ...
        training_history = self.q_net.fit(x=state_batch,                    # algo. line 15
                                          y=target_q,
                                          batch_size=batch_size,
                                          epochs=1,
                                          verbose=0)            # Gradient descent
        loss = training_history.history['loss']
        return loss[0]

    def test(self, _env, _no_of_episodes, _render=False):
        """
        Evaluates the performance of the current DQN agent by using it to play a
        few episodes of the game and then calculates the average reward it gets.
        The higher the average reward is the better the DQN agent performs.

        :param env: the game environment
        :param agent: the DQN agent
        :return: average reward across episodes
        """
        total_reward = 0.0
        episodes_to_play = _no_of_episodes
        for i in range(episodes_to_play):
            state = _env.reset()
            if _render:
                _env.render()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = _env.step(action)
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        episode_reward_avg = total_reward / episodes_to_play
        return episode_reward_avg

    def save_model(self, _folder):
        model_dir = _folder + "/model/"
        target_model_dir = _folder + "/target_model/"
        self.q_net.save(model_dir)
        self.target_q_net.save(target_model_dir)


def norm_state(_state):
    """
    Normalize 4 state vars.
    These norm values were calculated during first training jobs
    by observing max values over long runs.

    :param _state: array(4, ) contains
    :return: normalized state
    """
    _state[0] /= 2.4
    _state[1] /= 3.4
    _state[2] /= 0.22
    _state[3] /= 3.4
    return _state


def initialize_parameters(_env):
    _params = {
        'env_name': _env.unwrapped.spec.id, 'state_size': _env.observation_space.shape[0],
        'action_size': _env.action_space.n, 'max_steps': _env._max_episode_steps,
        'ttl_episode': 1_200, 'learning_rate': 0.0005,
        'gamma': 0.95, 'min_eps': 0.001, 'eps_decay_rate': 0.99,
        'epsilon': 1, 'batch_size': 64, 'C': 100, 'exp_replay_size': 100_000,
        'optimizer': 'RMSprop', 'initializer': "he_uniform",
        'layers': (32, 8)
    }
    return _params


def train_agent(env, params, logger, save=False):
    try:
        time_started = logger.time_started
        model_dir = logger.model_dir
        logger.write("\n\t\t---START TRAINING---\n")
        tensorboard = ModifiedTensorBoard(log_dir="logs_sec2_try2/{}-{}".format("train", time_started))

        avg_100_ep = deque(maxlen=100)

        memory_buffer = ExperienceReplay(params['exp_replay_size'], params['batch_size'])
        agent = NeuralNetwork(params['state_size'], params['action_size'],
                              params['learning_rate'], params['layers'],
                              params['optimizer'], params['initializer'])

        agent.q_net.summary()
        agent.target_q_net.summary()

        C_cnt = 1                                   # to apply algorithm line 16
        C = params['C']
        epsilon = params['epsilon']
        gamma = params['gamma']
        for episode in range(1, params['ttl_episode'] + 1):
            state = env.reset()
            state = norm_state(state)
            reward = 0
            for step in range(1, params['max_steps'] + 1):
                tensorboard.step = episode                          # TensorBoard graph update
                action = agent.greedy_e_policy(state, epsilon)      # algorithm line 5
                next_state, st_reward, done, _ = env.step(action)
                next_state = norm_state(next_state)
                reward += st_reward/500
                memory_buffer.store(state, action, reward, next_state, done)
                state = next_state
                minibatch = memory_buffer.sample_batch()
                loss = agent.train(minibatch, gamma)
                if C_cnt % C == 0:                  # algo. line 16
                    agent.update_target_network()
                C_cnt += 1
                if done:                            # on terminal step start over
                    break
            # end step loop
            if epsilon > params['min_eps']:
                epsilon *= params['eps_decay_rate']
            avg_100_ep.append(step)
            avg_reward = np.average(avg_100_ep)
            logger.write('Episode {}/{}, reward:{:3}, avg: {:.4}'
                         ' loss: {:.6}, eps: {:.4}\n'.format(episode, params['ttl_episode'],
                                                             step, avg_reward, loss, epsilon))
            tensorboard.update_stats(loss=loss,
                                     reward_steps=step,
                                     reward_avg100=avg_reward
                                     )
        # end episode loop
    finally:
        if save is True:
            agent.save_model(model_dir)

        env.close()


def test_agent(env, params, logger, episodes_to_play):
    agent = NeuralNetwork(params['state_size'], params['action_size'],
                          params['learning_rate'], params['layers'],
                          params['optimizer'], params['initializer'])
    performance = agent.test(env, episodes_to_play)
    logger.write("Testing training model:\n "
                 "Performance: {}, for {} episodes.\n".
                 format(performance, episodes_to_play))


def main():
    env = gym.make('CartPole-v1')
    params = initialize_parameters(env)

    params['layers'] = (8, 16)
    params['learning_rate'] = 0.001
    params['gamma'] = 0.95
    params['batch_size'] = 64
    # params['optimizer'] = 'Adam'

    params['optimizer'] = 'Adam'
    logger = Log("log.log", params)
    train_agent(env, params, logger, True)  # save=True, save model
    test_agent(env, params, logger, 100)

    params['optimizer'] = 'SGD'
    logger = Log("log.log", params)
    train_agent(env, params, logger, True)  # save=True, save model
    test_agent(env, params, logger, 100)

    logger.close()


if __name__ == '__main__':
    main()
