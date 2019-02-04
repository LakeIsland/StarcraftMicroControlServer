import socket
import pickle
import struct
from deep_q_agent import *
import datetime
from deep_sarsa_agent import *
from model_server import ModelServer


class DQNServer(ModelServer):
    def __init__(self):
        super().__init__()
        self.agent = None
        self.algorithm = 'deep_q'

    def waitForClient(self):
        super().wait_for_client()
        super().init_info()

        self.agent = DeepQAgent(action_size=self.action_size, state_size=self.state_size, layers=self.layers)
        if not (self.file_to_load == ''):
            self.agent.model.load_weights(self.file_to_load)
            self.agent.target_model.load_weights(self.file_to_load)

            print("read from", self.file_to_load)

        episode = 0

        while episode < self.max_iterate:
            while True:
                tag, msg = self.receiveMessage()
                if tag == "finish":
                    print("tag FINISHED")
                    break
                elif tag == "state":
                    action = self.agent.get_action(msg)
                    self.sendMessage("action", [action])
                elif tag == "sarsa":
                    sarsa = msg
                    self.agent.append_sample(self.npreshape(sarsa[0]), sarsa[1],
                                           sarsa[2], self.npreshape(sarsa[3]), sarsa[5])

                    if len(self.agent.memory) >= self.agent.train_start:
                        self.agent.train_model()

                    self.sendMessage("trainFinished", [1])
                else:
                    print("Unclassified tag", tag)

            episode += 1

            if(self.export_per != -1 and episode % self.export_per == 0):
                self.export(episode)

            self.agent.update_target_model()
            print("Memory size:", len(self.agent.memory))
            print("Episode %d ended." % (episode))

        self.c.close()
        if not self.evaluate:
            self.export(self.max_iterate)
