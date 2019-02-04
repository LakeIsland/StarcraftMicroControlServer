from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
import random
from deep_agent import *
import tensorflow.contrib.layers as tfcl
import keras.backend as k
import tensorflow as tf
#
# def get_weight_grad(model, inputs, outputs):
#     """ Gets gradient of model for given inputs and outputs for all weights"""
#     grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad

count = 0
class DeepSarsaAgent(DeepAgent):
    def __init__(self, state_size, action_size, layers):
        super().__init__(state_size, action_size, layers)

    def train_model(self, state, action, reward, next_state, next_action, done = 0):
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        #old_v = target[action]

        if done == 1:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                          self.model.predict(next_state)[0][next_action])

        #delta = target[action] - old_v

        target = np.reshape(target,[1,self.action_size])
        self.model.fit(state, target, epochs=1, verbose=0)

        # global count
        # count += 1
        # if count % 100 == 0:
        #
        #     outputTensor = self.model.output
        #     print(outputTensor.shape)
        #     listOfVariableTensors = self.model.trainable_weights
        #     gradients = k.gradients(outputTensor, listOfVariableTensors)
        #     #gradients  = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        #     sess = tf.InteractiveSession()
        #     sess.run(tf.initialize_all_variables())
        #     evaluated_gradients = sess.run(gradients, feed_dict={self.model.input: state})
        #
        #     for aa in evaluated_gradients:
        #         print('size', aa.shape)
        #         print(type(aa))
        #
        #     print('output_grad', evaluated_gradients[-1])
        #
        #     gra = get_weight_grad(self.model, state, target)
        #
        #     for aa in gra:
        #         print('size', aa.shape)
        #         print(type(aa))
        #     print('loss_grad', gra[-1])
        #
        #     print('delta', delta)


