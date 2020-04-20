import sys
import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class Network:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        #self.lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(lr,decay_steps=100000,decay_rate=0.96,staircase=True)
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)


        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)


            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def reward_sum(self, rewards,i,n_steps,gamma):
        total_reward=0
        for r in range(0,n_steps):
            #print("rewards {}:{}".format(r,rewards[i]*(gamma**r)))
            total_reward+=rewards[i+r]*(gamma**r)

        #print("total_reward:{}".format(total_reward))
        #exit()

        return total_reward

    def Multi_train(self, TargetNet,n_steps):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=max(len(self.experience['s'])-n_steps+1,0), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([ self.reward_sum(self.experience['r'],i,n_steps,self.gamma) for i in ids])
        states_n_step = np.asarray([self.experience['s2'][i+n_steps-1] for i in ids])
        dones = np.asarray([self.experience['done'][i+n_steps-1] for i in ids])
        value_n_step = np.max(TargetNet.predict(states_n_step), axis=1)
        actual_values = np.where(dones, rewards, rewards+(self.gamma**n_steps)*value_n_step)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)


            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)


class DQN:
    def __init__(self, env, multistep=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.n_steps = 3            # Multistep(n-step) 구현 시 n 값, 수정 가능

    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)

        if self.multistep==False:
            gamma = 0.99
            copy_step = 3
            num_states = len(self.env.observation_space.sample())
            num_actions = self.env.action_space.n
            hidden_units = [200, 200]
            max_experiences = 10000
            min_experiences = 100
            batch_size = 32
            lr = 1e-4
        else:
            gamma = 0.99
            copy_step = 3
            num_states = len(self.env.observation_space.sample())
            num_actions = self.env.action_space.n
            hidden_units = [200, 200]
            max_experiences = 10000
            min_experiences = 100
            batch_size = 32
            lr = 1e-4

        TrainNet = Network(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        TargetNet = Network(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

        total_rewards = np.empty(max_episode)

        epsilon = 0.1
        decay = 0.9999




        for episode in range(max_episode):
            iter=0
            rewards = 0
            done = False
            state = self.env.reset()
            step_count = 0
            observations = self.env.reset()
            losses = list()
            epsilon = epsilon * decay


            # episode 시작
            while not done:
                action = TrainNet.get_action(observations, epsilon)
                prev_observations = observations
                observations, reward, done, _ = self.env.step(action)
                rewards += reward
                if done:
                    reward = -250
                    self.env.reset()

                exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
                TrainNet.add_experience(exp)

                if self.multistep==False:
                    loss = TrainNet.train(TargetNet)
                else:
                    loss = TrainNet.Multi_train(TargetNet, self.n_steps)

                if isinstance(loss, int):
                    losses.append(loss)
                else:
                    losses.append(loss.numpy())
                iter += 1
                if iter % copy_step == 0:
                    TargetNet.copy_weights(TrainNet)




                #state = next_state
                step_count += 1
            #
            # avg_loss=np.mean(losses)
            # total_rewards[episode] = rewards
            # avg_rewards = total_rewards[max(0, episode - 100):(episode + 1)].mean()
            #
            #
            # print("episode reward:{}".format(rewards))
            # print("running avg reward(100):{}".format(avg_rewards))
            # print("average loss:{}".format(avg_loss))

            last_100_episode_step_count.append(step_count)



            avg_step_count = np.mean(last_100_episode_step_count)
            print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))

            avg_step_count_list.append(avg_step_count)

            if avg_step_count>=475:
                break

        return avg_step_count_list

