from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
import random
import keras.backend as K

class DeepAgent:
    def __init__(self, state_size, action_size, layers):
        self.load_model = False
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001

        self.model = self.build_model(layers)

    def build_model(self, layers):
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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, msg):
        state = np.reshape(msg, [1, self.state_size])
        own_unit = msg[10:18]
        ene_unit = msg[26:34]
        terrains = msg[34:42]

        q_values = self.model.predict(state)
        kk = q_values[0]
        for i in range(8):
            kk[i] = kk[i] if terrains[i] < 0.7 and own_unit[i] < 0.7 and ene_unit[i] < 0.7 else np.NINF

        action = int(np.argmax(kk))
        return action