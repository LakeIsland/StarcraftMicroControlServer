from model_server import ModelServer
from rl_agent.deep_sarsa_agent import *
from rl_agent.a2c_agent import *
import os

class DeepSARSAServer(ModelServer):
    def __init__(self, port=1234):
        super().__init__(port)
        self.agent = None

    def waitForClient(self):
        super().wait_for_client()
        super().init_info()

        if self.algorithm == 'DeepSarsa':
            self.agent = DeepSarsaAgent(action_size=self.action_size, state_size=self.state_size, layers=self.layers, use_eligibility_trace=self.eligibility_trace)
        elif self.algorithm == 'A2C':
            self.agent = A2CAgent(state_size=self.state_size, action_size=self.action_size, actor_layers=self.actor_layers,
                                  critic_layers=self.critic_layers)

        if self.file_or_folder_to_load != '':
            if os.path.isfile(self.file_or_folder_to_load):
                self.agent.model.load_weights(self.file_or_folder_to_load)
                print("read from", self.file_or_folder_to_load)

        while True:
            tag, msg = self.receiveMessage()
            if tag == "state":
                action = self.agent.get_action(msg)
                self.sendMessage("action", [action])
            elif tag == "sarsa":
                sarsa = msg
                self.agent.train_model(self.npreshape(sarsa[0]), sarsa[1],
                                       sarsa[2], self.npreshape(sarsa[3]), sarsa[4], sarsa[5], sarsa[6])
                self.sendMessage("train finished", [1])

            elif tag == "export":
                episode = msg[0]
                print("Export at %d"% episode)
                self.export(episode)
                self.sendMessage(tag="export finished", msg=[11111])

            elif tag == "episode finished":
                episode = msg[0]
                print("Episode %d ended." % episode)
                if self.eligibility_trace:
                    self.agent.clear_eligibility_records()

            elif tag == "load_file":
                if self.load_file_counter < len(self.load_files_list):
                    episode_count, file_to_load = self.load_files_list[self.load_file_counter]
                    self.agent.model.load_weights(self.file_or_folder_to_load+ '/'+file_to_load)

                    print("%d trained model loaded:" % episode_count, file_to_load)
                    self.sendMessage(tag="load finished", msg=[episode_count, file_to_load])
                    self.load_file_counter += 1
                else:
                    self.sendMessage(tag="no more file to load", msg=[11111])
                    break
            elif tag == "end connection":
                print("Connection Ended")
                break

            elif tag == "test_multiple_model_info":
                test_file_path = msg['test_file_path']
                test_per = msg['test_per']
                self.set_files_to_load(model_folder=test_file_path, test_per=test_per)
                self.sendMessage(tag="file_number", msg=len(self.load_files_list))
            else:
                print("Unclassified tag", tag)
        #
        # episode = 0
        # while episode <= self.max_iterate:
        #     while True:
        #         tag, msg = self.receiveMessage()
        #         if tag == "finish":
        #             print("tag FINISHED")
        #             break
        #         elif tag == "state":
        #             action = self.agent.get_action(msg)
        #             self.sendMessage("action", [action])
        #         elif tag == "sarsa":
        #             sarsa = msg
        #             self.agent.train_model(self.npreshape(sarsa[0]), sarsa[1],
        #                                    sarsa[2], self.npreshape(sarsa[3]), sarsa[4], sarsa[5], sarsa[6])
        #             self.sendMessage("trainFinished", [1])
        #         else:
        #             print("Unclassified tag", tag)
        #
        #     if episode == self.max_iterate:
        #         break
        #     episode += 1
        #     print("Episode %d ended." % (episode))
        #
        #     if(self.eligibility_trace):
        #         self.agent.clear_eligibility_records()
        #
        #     if(self.export_per != -1 and episode % self.export_per == 0):
        #         self.export(episode)

        self.c.close()

