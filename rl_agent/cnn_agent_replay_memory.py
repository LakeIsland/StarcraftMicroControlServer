from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D
import keras
from keras.models import Model
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from collections import deque
import numpy as np
import tensorflow as tf

import random
import keras.backend as K
from rl_agent.rl_agent import RLAgent


class CNNAgentWithReplay(RLAgent):
    def __init__(self, frame_size, minimap_frame_size, non_spatial_state_size, action_size):
        self.load_model = False
        self.action_size = action_size
        self.frame_size = frame_size
        self.minimap_frame_size = minimap_frame_size

        self.non_spatial_state_size = non_spatial_state_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.batch_size = 32
        self.train_start = 1000
        self.train_started = False

        self.update_target_model_counter = 0
        self.update_target_model_period = 10

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.memory = deque(maxlen=10000)

    def build_model(self):

        local_map_input = Input(shape=self.frame_size)
        conv1 = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu')(local_map_input)
        conv1 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv1)
        conv1 = Flatten()(conv1)

        minimap_map_input = Input(shape=self.minimap_frame_size)
        conv2 = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu')(minimap_map_input)
        conv2 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv2)
        conv2 = Flatten()(conv2)

        non_spatial_input = Input(shape=(self.non_spatial_state_size, ))

        x = keras.layers.concatenate([conv1, conv2, non_spatial_input])
        x = Dense(512, activation='relu')(x)
        main_output = Dense(self.action_size, activation='linear')(x)

        # local_map_model.add(Convolution2D(32, kernel_size=(8, 8), strides=(4, 4), input_shape=self.frame_size, activation='relu'))
        # local_map_model.add(Convolution2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        # local_map_model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        # local_map_model.add(Flatten())
        # local_map_model.add(Dense(512, activation='relu'))
        #
        # local_map_model.add(Dense(self.action_size, activation='linear'))

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

        model = Model(inputs=[local_map_input, minimap_map_input, non_spatial_input], outputs=main_output)

        model.summary()
        o = RMSprop(lr=0.00025, epsilon=0.01)
        model.compile(loss='mse', optimizer=o)
        return model

    def update_target_model(self):
        self.update_target_model_counter += 1
        if self.update_target_model_counter == self.update_target_model_period:
            print("Target Updated")
            self.update_target_model_counter = 0
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self, msg):
        state_spatial = np.reshape(msg[0], (1,) + self.frame_size)
        state_minimap = np.reshape(msg[1], (1,) + self.minimap_frame_size)
        state_non_spatial = np.reshape(msg[2], (1, self.non_spatial_state_size))
        actions = self.model.predict([state_spatial, state_minimap, state_non_spatial])[0]

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

    def append_sample(self, state, action, reward, next_state, next_action, done=0):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def train_model(self, state, action, reward, next_state, next_action, unit_id, done = 0):

        self.append_sample(state, action, reward, next_state, next_action, done)

        if len(self.memory) >= self.train_start:
            if not self.train_started:
                self.train_started = True
                print("Train started")
            mini_batch = random.sample(self.memory, self.batch_size)

            states_spatial = np.zeros((self.batch_size,) + self.frame_size)
            states_non_spatial = np.zeros((self.batch_size, self.non_spatial_state_size))
            states_minimap = np.zeros((self.batch_size,) + self.minimap_frame_size)

            next_states_spatial = np.zeros((self.batch_size,) + self.frame_size)
            next_states_non_spatial = np.zeros((self.batch_size, self.non_spatial_state_size))
            next_states_minimap = np.zeros((self.batch_size,) + self.minimap_frame_size)

            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                states_spatial[i] = mini_batch[i][0][0].reshape(self.frame_size)
                states_minimap[i] = mini_batch[i][0][1].reshape(self.minimap_frame_size)
                states_non_spatial[i] = mini_batch[i][0][2]

                next_states_spatial[i] = mini_batch[i][3][0].reshape(self.frame_size)
                next_states_minimap[i] = mini_batch[i][3][1].reshape(self.minimap_frame_size)
                next_states_non_spatial[i] = mini_batch[i][3][2]

                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                dones.append(mini_batch[i][4])

            targets = self.model.predict([states_spatial, states_minimap, states_non_spatial])
            next_values = self.target_model.predict([next_states_spatial, next_states_minimap, next_states_non_spatial])

            for i in range(self.batch_size):
                done = dones[i]
                if done == 1:
                    targets[i][action] = reward
                else:
                    targets[i][action] = (reward + self.discount_factor * np.amax(next_values[i][0]))

            self.model.fit([states_spatial, states_minimap, states_non_spatial], targets, epochs=1, verbose=0)
