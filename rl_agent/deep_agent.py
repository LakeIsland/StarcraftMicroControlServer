from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Sequential
import numpy as np
import tensorflow as tf

import random
import keras.backend as K
from rl_agent.rl_agent import RLAgent


class DeepAgent(RLAgent):
    def __init__(self, state_size, action_size, layers, use_eligibility_trace = False):
        self.load_model = False
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.eligibility_trace_lambda = 0.8
        self.use_eligibility_trace = use_eligibility_trace
        self.model = self.build_model(layers, use_eligibility_trace)

    def build_model(self, layers, use_eligibility_trace):
        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                model.add(Dense(layers[i], input_dim=self.state_size, activation='relu',
                                kernel_initializer='he_uniform'))
            else:
                model.add(Dense(layers[i], activation='relu',
                                kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        if use_eligibility_trace:
            print("USE absoulute error and SGD")

            def custom_loss(y_true, y_pred):
                return K.sum(y_pred, axis = -1)

            model.compile(loss='mean_absolute_error', optimizer=SGD(lr=self.learning_rate))
        else:
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, msg):
        state = np.reshape(msg, [1, self.state_size])
        actions = self.model.predict(state)[0]
        #actions = self.predict_with_check_validity(msg)
        action = int(np.argmax(actions))
        return action

    def predict_with_check_validity(self, msg):
        state = np.reshape(msg, [1, self.state_size])
        own_unit = msg[10:18]
        ene_unit = msg[26:34]
        terrains = msg[34:42]

        q_values = self.model.predict(state)
        kk = q_values[0]
        for i in range(8):
            kk[i] = np.NINF if terrains[i] > 0.7 or own_unit[i] > 0.7 or ene_unit[i] > 0.7 else kk[i]
        return kk

    def save_model(self, file_name):
        self.model.save_weights(file_name)