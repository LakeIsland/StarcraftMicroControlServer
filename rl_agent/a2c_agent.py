from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Sequential
import numpy as np
import tensorflow as tf

import random
import keras.backend as K
from rl_agent.rl_agent import RLAgent


class A2CAgent(RLAgent):
    def __init__(self, state_size, action_size, actor_layers, critic_layers):
        self.load_model = False
        self.action_size = action_size
        self.state_size = state_size

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor(actor_layers)
        self.critic = self.build_critic(critic_layers)
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()


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

    def build_critic(self, critic_layers):
        critic = Sequential()

        for i, n in enumerate(critic_layers):
            if i == 0:
                critic.add(Dense(n, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
            else:
                critic.add(Dense(n, activation='relu', kernel_initializer='he_uniform'))

        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))

        critic.summary()
        return critic

    def get_action(self, msg):
        state = np.reshape(msg, [1, self.state_size])
        policy = self.actor.predict(state, batch_size=1).flatten()
        return int(np.random.choice(self.action_size, 1, p=policy)[0])

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    def train_model(self, state, action, reward, next_state, next_action, unit_id, done=0):

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])
