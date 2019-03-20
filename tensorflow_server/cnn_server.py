from model_server import ModelServer
from rl_agent.deep_sarsa_agent import *
from rl_agent.eligibility_a2c_agent import *
import os
from rl_agent.cnn_agent import *
from rl_agent.cnn_agent_replay_memory import *
from rl_agent.cnn_agent_eligibility import *


class CNNServer(ModelServer):
    def __init__(self, port=1234):
        super().__init__(port)
        self.agent = None

    def waitForClient(self):
        super().wait_for_client()
        super().init_info()

        #self.agent = CNNAgentWithReplay(frame_size=self.frame_size, minimap_frame_size=self.minimap_frame_size, non_spatial_state_size=self.non_spatial_state_size, action_size=self.action_size)
        self.agent = CNNAgentWithReplayEligibility(frame_size=self.frame_size,
            minimap_frame_size=self.minimap_frame_size, non_spatial_state_size=self.non_spatial_state_size,
            action_size=self.action_size)
        if self.file_or_folder_to_load != '':
            if os.path.isfile(self.file_or_folder_to_load):
                self.agent.model.load_weights(self.file_or_folder_to_load)
                print("read from", self.file_or_folder_to_load)

        while True:
            tag, msg = self.receiveMessage()
            #last_minimap_state = None
            #minimap_state = None
            if tag == "state":
                s_spatial = np.frombuffer(msg[0], dtype=np.float32)
                s_minimap = np.frombuffer(msg[1], dtype=np.float32)
                s_non_spatial = np.array(msg[2], dtype=np.float32)
                action = self.agent.get_action((s_spatial, s_minimap, s_non_spatial))
                self.sendMessage("action", [action])
            elif tag == "sarsa":
                sarsa = msg
                s_spatial = np.frombuffer(sarsa[0][0], dtype=np.float32)
                s_minimap = np.frombuffer(sarsa[0][1], dtype=np.float32)
                s_non_spatial = np.array(sarsa[0][2], dtype=np.float32)

                ns_spatial = np.frombuffer(sarsa[3][0], dtype=np.float32)
                ns_minimap = np.frombuffer(sarsa[3][1], dtype=np.float32)
                ns_non_spatial = np.array(sarsa[3][2], dtype=np.float32)

                #s2 = np.frombuffer(sarsa[3], dtype=np.float32)
                self.agent.train_model((s_spatial, s_minimap, s_non_spatial),
                                       sarsa[1],
                                       sarsa[2],
                                       (ns_spatial, ns_minimap, ns_non_spatial), sarsa[4], sarsa[5], sarsa[6])
                self.sendMessage("train finished", [1])

            elif tag == "export":
                episode = msg[0]
                print("Export at %d"% episode)
                self.export(episode)
                self.sendMessage(tag="export finished", msg=[11111])

            elif tag == "episode finished":
                episode = msg[0]
                #self.agent.update_target_model()
                #print("Episode %d ended. Memory Size %d" % (episode, len(self.agent.memory)) )
                print("Episode %d ended." % episode)
                self.agent.clear_eligibility_record()

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

