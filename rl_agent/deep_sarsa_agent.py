from rl_agent.deep_agent import *
import keras.backend as k
import tensorflow as tf

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

count = 0
class DeepSarsaAgent(DeepAgent):
    def __init__(self, state_size, action_size, layers, use_eligibility_trace):
        super().__init__(state_size, action_size, layers, use_eligibility_trace)
        self.eligibility_trace_records = {}

    def train_with_eligibility_trace(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        q_value = target[action]

        if done == 1:
            expected_q_value = reward
        else:
            expected_q_value = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        delta = expected_q_value - q_value

        weights = np.array(self.model.get_weights())

        outputTensor = self.model.output[:, action]
        listOfVariableTensors = self.model.trainable_weights
        gradients = k.gradients(outputTensor, listOfVariableTensors)

        evaluated_gradients = self.sess.run(gradients, feed_dict={self.model.input: state})
        evaluated_gradients = np.array(evaluated_gradients)

        e_t1 = self.eligibility_trace_records.get(unit_id, None)
        e_t = evaluated_gradients
        if e_t1 is not None:
            e_t += self.discount_factor * self.eligibility_trace_lambda * e_t1

        self.eligibility_trace_records[unit_id] = e_t

        new_weight = weights + self.learning_rate * delta * e_t
        self.model.set_weights(new_weight)

    def train_with_eligibility_trace2(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        old_v = target[action]

        if done == 1:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])
        #
        delta = target[action] - old_v

        target = np.reshape(target, [1, self.action_size])
        before_weight = np.array(self.model.get_weights())
        self.model.fit(state, target, epochs=1, verbose=0)
        after_weight = np.array(self.model.get_weights())

        difference = after_weight - before_weight

        e_t1 = self.eligibility_trace_records.get(unit_id, None)
        e_t = difference
        if e_t1 is not None:
            e_t += self.eligibility_trace_lambda * e_t1

        self.eligibility_trace_records[unit_id] = e_t

        new_weight = before_weight + e_t
        self.model.set_weights(new_weight)

    def train_with_eligibility_trace3(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        q_value = target[action]

        if done == 1:
            expected_q_value = reward
        else:
            expected_q_value = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        delta = expected_q_value - q_value

        weights = np.array(self.model.get_weights())

        grads = self.model.optimizer.get_gradients(self.model.output[:, action], self.model.trainable_weights)
        f = K.function([self.model.layers[0].input], grads)
        output_grad = f([state])

        evaluated_gradients = np.array(output_grad)

        e_t1 = self.eligibility_trace_records.get(unit_id, None)
        e_t = evaluated_gradients
        if e_t1 is not None:
            e_t += self.discount_factor * self.eligibility_trace_lambda * e_t1

        self.eligibility_trace_records[unit_id] = e_t

        new_weight = weights + self.learning_rate * delta * e_t
        self.model.set_weights(new_weight)

    def train_with_eligibility_trace4(self, state, action, reward, next_state, next_action, unit_id, done = 0):
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        old_q = target[action]
        if done == 1:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        delta = target[action] - old_q

        before_weights = np.array(self.model.get_weights())

        target = np.reshape(target, [1, self.action_size])
        self.model.fit(state, target, epochs=1, verbose=0)

        after_weights = np.array(self.model.get_weights())

        difference = after_weights - before_weights

        gradient_q = difference * (self.action_size / self.learning_rate * (1 if delta > 0 else -1))

        e_t_prev = self.eligibility_trace_records.get(unit_id, None)
        e_t = gradient_q
        if e_t_prev is not None:
            e_t += self.discount_factor * self.eligibility_trace_lambda * e_t_prev

        self.eligibility_trace_records[unit_id] = e_t

        new_weight = before_weights + self.learning_rate * delta * e_t
        self.model.set_weights(new_weight)

    def clear_eligibility_records(self):
        self.eligibility_trace_records.clear()

    def train_model(self, state, action, reward, next_state, next_action, unit_id, done = 0):

        if self.use_eligibility_trace:
            self.train_with_eligibility_trace4(state, action, reward, next_state, next_action, unit_id, done)
            return

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        if done == 1:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                          self.model.predict(next_state)[0][next_action])
        target = np.reshape(target,[1,self.action_size])
        self.model.fit(state, target, epochs=1, verbose=0)

    def train_model_debug(self, state, action, reward, next_state, next_action, unit_id, done = 0):

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        old_v = target[action]

        if done == 1:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                          self.model.predict(next_state)[0][next_action])
        #
        delta = target[action] - old_v

        target = np.reshape(target,[1,self.action_size])

        #
        #
        #
        # weights = np.array(self.model.get_weights())
        #
        # outputTensor = self.model.output[:, action]
        # listOfVariableTensors = self.model.trainable_weights
        # gradients = k.gradients(outputTensor, listOfVariableTensors)
        # sess = tf.InteractiveSession()
        # sess.run(tf.initialize_all_variables())
        # evaluated_gradients = sess.run(gradients, feed_dict={self.model.input: state})
        #
        # e_t = self.discount_factor * self.eligibility_trace_lambda * e_t1
        # e_t += evaluated_gradients
        #
        # new_weight = weights + self.learning_rate * delta * e_t
        # self.model.set_weights(new_weight)
        #
        #
        #
        global count
        count += 1
        if count % 100 == 0:
            outputTensor = self.model.output[:,action]
            print(outputTensor)
            print(action)
            print(outputTensor.shape, "output shape")
            print(self.model.loss_weights)
            listOfVariableTensors = self.model.trainable_weights
            gradients = k.gradients(outputTensor, listOfVariableTensors)
            sess = tf.InteractiveSession()
            sess.run(tf.initialize_all_variables())
            evaluated_gradients = sess.run(gradients, feed_dict={self.model.input: state})

            # for aa in evaluated_gradients:
            #     print('size', aa.shape)
            #     print(type(aa))
            print('sess method', evaluated_gradients[-1])

            gra = get_weight_grad(self.model, state, target)

            # for aa in gra:
            #     print('size', aa.shape)
            #     print(type(aa))
            print('k method', gra[-1])

            sess.close()
            before_weight = np.array(self.model.get_weights())

            grads = self.model.optimizer.get_gradients(self.model.output[:, action], self.model.trainable_weights)
            f = K.function([self.model.layers[0].input], grads)
            output_grad = f([state])
            # for aa in output_grad:
            #     print('size', aa.shape)
            #     print(type(aa))
            print('k method own', output_grad[-1])

            print('delta', delta)


        self.model.fit(state, target, epochs=1, verbose=0 )
        #self.model.train_on_batch(state, target)

        if count % 100 == 0:
            after_weight = np.array(self.model.get_weights())
            diff = after_weight - before_weight
            print("DIFFERENCE")
            print(diff[-1] / (-self.learning_rate / 9 * (-1 if delta > 0 else 1)) - output_grad[-1])
            print(diff[-2] / (-self.learning_rate / 9 * (-1 if delta > 0 else 1)) - output_grad[-2])

            #print(diff[-1] / (-self.learning_rate) - output_grad[-1])
            #print(diff[-2] / (-self.learning_rate) - output_grad[-2])

            #print(diff[-1] / output_grad[-1]/(2 * self.learning_rate * delta / 9))
            #print(diff[-2] / output_grad[-2]/(2 * self.learning_rate * delta / 9))
