from model_server import ModelServer
import pickle
from deep_sarsa_agent import *
from deep_q_agent import *
import datetime

class DeepSARSAServer(ModelServer):
    def __init__(self):
        super().__init__()
        self.agent = None
        self.algorithm = 'deep_sarsa'

    def waitForClient(self):
        super().wait_for_client()
        super().init_info()
        self.agent = DeepSarsaAgent(action_size=self.action_size, state_size=self.state_size, layers=self.layers)

        if not (self.file_to_load == ''):
            self.agent.model.load_weights(self.file_to_load)
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
                    self.agent.train_model(self.npreshape(sarsa[0]), sarsa[1],
                                           sarsa[2], self.npreshape(sarsa[3]), sarsa[4], sarsa[5])
                    self.sendMessage("trainFinished", [1])
                else:
                    print("Unclassified tag", tag)

            episode += 1
            print("Episode %d ended." % (episode))
            if(self.export_per != -1 and episode % self.export_per == 0):
                self.export(episode)

        self.c.close()
        if not self.evaluate:
            self.export(self.max_iterate)

