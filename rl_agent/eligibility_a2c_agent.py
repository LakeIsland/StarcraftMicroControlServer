from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Sequential
import numpy as np
import tensorflow as tf

import random
import keras.backend as K
from rl_agent.rl_agent import RLAgent


def relu_plus_one(x):
    return K.relu(x) + 0.01


class A2CAgent(RLAgent):
    def __init__(self, state_size, action_size, actor_layers, critic_layers, use_eligibility_trace):
        self.load_model = False
        self.action_size = action_size
        self.state_size = state_size

        self.discount_factor = 0.9
        self.eligibility_trace_lambda = 0.8

        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.actor = self.build_actor(actor_layers)
        self.use_eligibility_trace = use_eligibility_trace
        self.critic = self.build_critic(critic_layers, use_eligibility_trace)
        self.critic_target = self.build_critic(critic_layers, use_eligibility_trace)

        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_updater = self.actor_optimizer()

        self.eligibility_trace_records = {}

    def build_actor(self, actor_layers):
        actor = Sequential()

        for i, n in enumerate(actor_layers):
            if i == 0:
                actor.add(Dense(n, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
            else:
                actor.add(Dense(n, activation='relu', kernel_initializer='he_uniform'))

        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))

        actor.summary()
        return actor

    def build_critic(self, critic_layers, use_eligibility_trace):
        critic = Sequential()

        for i, n in enumerate(critic_layers):
            if i == 0:
                critic.add(Dense(n, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
            else:
                critic.add(Dense(n, activation='relu', kernel_initializer='he_uniform'))

        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()

        if use_eligibility_trace:
            print("USE absoulute error and SGD")
            critic.compile(loss='mean_absolute_error', optimizer=SGD(lr=self.critic_lr))
        else:
            critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))

        return critic

    def get_action(self, msg):
        state = np.reshape(msg, [1, self.state_size])
        policy = self.actor.predict(state, batch_size=1).flatten()
        policy = policy / policy.sum()
        action = int(np.random.choice(self.action_size, 1, p=policy)[0])
        # if np.argmax(policy) == 8:
        #     print(policy)
        # if random.random() < 0.001:
        #     print(policy)

        return action

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        #p = self.actor.output / K.sum(self.actor.output, axis=1)
        #action_prob = K.sum(action * p, axis=1)
        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob + 1e-50) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    def train_critic(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        value = self.critic.predict(state)[0]
        if np.isnan(value):
            raise ValueError
        if done == 1:
            next_value = reward
        else:
            next_value = (reward + self.discount_factor * self.critic_target.predict(next_state)[0])
        if np.isnan(next_value):
            raise ValueError

        delta = next_value - value

        if np.isnan(delta):
            raise ValueError

        before_weights = np.array(self.critic.get_weights())
        for i in range(6):
            if np.isnan(before_weights[i]).any():
                raise ValueError

        self.critic.fit(np.float32(state), np.reshape(next_value, [1, 1]), batch_size=1,  epochs=1, verbose=0)
        after_weights = np.array(self.critic.get_weights())

        difference = after_weights - before_weights

        for i in range(6):
            if np.isnan(after_weights[i]).any():
                raise ValueError
            if np.isnan(difference[i]).any():
                raise ValueError

        gradient_v = difference * (1 / self.critic_lr * (1 if delta > 0 else -1))

        e_t_prev = self.eligibility_trace_records.get(unit_id, None)
        e_t = gradient_v
        if e_t_prev is not None:
            e_t += self.discount_factor * self.eligibility_trace_lambda * e_t_prev

        self.eligibility_trace_records[unit_id] = e_t

        new_weight = before_weights + self.critic_lr * delta * e_t

        self.critic.set_weights(new_weight)

    def clear_eligibility_records(self):
        self.eligibility_trace_records.clear()
        self.critic_target.set_weights(self.critic.get_weights())

    def train_model(self, state, action, reward, next_state, next_action, unit_id, done=0):
        value = self.critic.predict(state)[0]
        if np.isnan(value):
            raise ValueError
        next_value = self.critic_target.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])

        if self.use_eligibility_trace:
            self.train_critic(state, action, reward, next_state, next_action, unit_id, done)
        else:
            self.critic.fit(state, target, batch_size=1, verbose=0)

    def save_model(self, file_name):
        self.actor.save_weights(file_name+"_actor")
        self.critic.save_weights(file_name + "_critic")
