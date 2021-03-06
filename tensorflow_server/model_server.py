import socket
import pickle
import datetime
import numpy as np
import os

class ModelServer:
    def __init__(self, port=1234):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("소켓 생성완료")

        HOST = 'localhost'
        s.bind((HOST, port))
        s.listen(5)
        print("%d 포트로 연결을 기다리는중" % (port))

        self.s = s
        self.c = None
        self.agent = None
        self.algorithm = None
        self.mode = None

        now = datetime.datetime.now()
        self.created_time = now.strftime('%Y_%m_%d_%H_%M')

    def wait_for_client(self):
        while True:
            c, addr = self.s.accept()
            print(addr, "사용자가 접속함")
            self.c = c
            break

    def init_info(self):
        tag, init_info = self.receiveMessage()
        assert tag == "init_info"

        self.mode = init_info['mode']
        self.max_iterate = init_info['max_iterate']
        self.file_or_folder_to_load = init_info.get('file_or_folder_to_load', '')

        self.action_size = init_info.get('action_size')
        self.state_size = init_info.get('state_size')
        self.frame_size = init_info.get('frame_size', None)
        self.minimap_frame_size = init_info.get('minimap_frame_size', None)

        self.non_spatial_state_size = init_info.get('non_spatial_state_size', 0)

        self.layers = init_info.get('layers', None)

        self.actor_layers = init_info.get('actor_layers', None)
        self.critic_layers = init_info.get('critic_layers', None)

        self.map_name = init_info.get('map_name')
        self.export_per = init_info.get('export_per', -1)

        self.eligibility_trace = init_info.get('eligibility_trace', False)

        self.algorithm = init_info.get('algorithm', None)

        print("Algorithm:", self.algorithm)
        print('Layer size:', self.layers)
        print("Mode : ", self.mode)
        print("map name: ", self.map_name)
        print(self.max_iterate, " Iterate")
        print("export_per:", self.export_per)
        print("Eligibility Trace:", self.eligibility_trace)

        self.evaluate = self.mode != 'train'
        self.load_files_list = []
        self.load_file_counter = 0

        self.sendMessage(tag='init finished', msg=[11111])

    def set_files_to_load(self, model_folder, test_per):
        for f in os.listdir("../modeldata/"+model_folder):
            file_name_split = f.split('_')
            episode_count = int(file_name_split[-7])
            if test_per == -1:
                self.load_files_list.append((episode_count, f))
            elif episode_count % test_per == 0:
                self.load_files_list.append((episode_count, f))
        self.load_files_list.sort(key=lambda x: x[0])
        for f in self.load_files_list:
            print(f)
        self.file_or_folder_to_load = "../modeldata/"+model_folder

    def export(self, n_iterate):

        directory = "../modeldata/%s_%s_%s" % (self.algorithm, self.map_name, self.created_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        now = datetime.datetime.now()
        nowDate = now.strftime('%Y_%m_%d_%H_%M')
        self.agent.save_model("%s/%s_%s_%d_times_%s"%(directory, self.algorithm, self.map_name, n_iterate, nowDate))

        print("EXPORTED")

    def sendMessage(self, tag, msg):
        packet = pickle.dumps([tag, msg])
        self.c.send(packet)
        #print("SEND", tag, msg)

    def receiveMessage(self):
        data = b""
        while True:
            packet = self.c.recv(4096)
            if not packet:
                break
            #print("RECEIVED")
            data += packet
            if len(packet) < 4096: break

        data = pickle.loads(data)
        tag = data[0]
        msg = data[1]
        #print("RECEIVE", tag, msg)
        return tag, msg

    def npreshape(self, l):
        return np.reshape(l, [1, len(l)])