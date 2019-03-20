from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

import numpy as np
import tensorflow as tf

import random
import keras.backend as K
from rl_agent.rl_agent import RLAgent


class CNNAgent(RLAgent):
    def __init__(self, frame_size, action_size):
        self.load_model = False
        self.action_size = action_size
        self.frame_size = frame_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, kernel_size=(8, 8), strides=(4, 4), input_shape=self.frame_size, activation='relu'))
        model.add(Convolution2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        #model.add(MaxPooling2D(pool_size=2))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))

        # model = Sequential()
        # model.add(Convolution2D(16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.frame_size, activation='relu'))
        # model.add(Convolution2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(9, activation='linear'))

        # model.add()
        # for i in range(len(layers)):
        #     if i == 0:
        #         model.add(Dense(layers[i], input_dim=self.state_size, activation='relu',
        #                         kernel_initializer='he_uniform'))
        #     else:
        #         model.add(Dense(layers[i], activation='relu',
        #                         kernel_initializer='he_uniform'))
        # model.add(Dense(self.action_size, activation='linear',
        #                 kernel_initializer='he_uniform'))

        model.summary()
        o = RMSprop(lr=0.00025, epsilon=0.01)
        model.compile(loss='mse', optimizer=o)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, msg):
        state = np.reshape(msg, (1, ) + self.frame_size)
        actions = self.model.predict(state)[0]
        action = int(np.argmax(actions))
        return action

    def save_model(self, file_name):
        name=file_name
        if not file_name.endswith(".h5"):
            name += ".h5"
        self.model.save_weights(name)

    def load_model(self, file_name):
        print(file_name, "loaded")
        self.model.load_weights(file_name)

    def train_model(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        state2 = np.reshape(state, (1, )+ self.frame_size)
        target = self.model.predict(state2)[0]
        if done == 1:
            target[action] = reward
        else:
            next_state2 = np.reshape(next_state, (1,) + self.frame_size)
            target[action] = (reward + self.discount_factor *
                          self.target_model.predict(next_state2)[0][next_action])
        target = np.reshape(target,[1,self.action_size])
        self.model.fit(state2, target, epochs=1, verbose=0)