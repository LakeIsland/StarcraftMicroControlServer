import socket
import pickle
import datetime
import numpy as np

class ModelServer:
    def __init__(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("소켓 생성완료")

        HOST = 'localhost'
        port = 1234
        s.bind((HOST, port))
        s.listen(5)
        print("%d 포트로 연결을 기다리는중" % (port))

        self.s = s
        self.c = None
        self.agent = None
        self.algorithm = None
        self.mode = None

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
        self.file_to_load = init_info.get('file_to_load', '')

        self.action_size = init_info.get('action_size')
        self.state_size = init_info.get('state_size')
        self.layers = init_info.get('layers')
        self.map_name = init_info.get('map_name')
        self.export_per = init_info.get('export_per', -1)


        print('Layer size:', self.layers)
        print("Mode : ", self.mode)
        print("map name: ", self.map_name)
        print(self.max_iterate, " Iterate")
        print("export_per:", self.export_per)

        self.evaluate = self.mode == 'evaluate'

    def export(self, n_iterate):
        now = datetime.datetime.now()
        nowDate = now.strftime('%Y_%m_%d_%H_%M')
        self.agent.model.save_weights("../modeldata/%s_%s_%d_times_%s.h5"%(self.algorithm, self.map_name, n_iterate, nowDate))
        print("EXPORTED")

    def sendMessage(self, tag, msg):
        packet = pickle.dumps([tag, msg])
        self.c.send(packet)
        #print("SEND", tag, msg)

    def receiveMessage(self):
        data = pickle.loads(self.c.recv(1024))
        tag = data[0]
        msg = data[1]
        #print("RECEIVE", tag, msg)
        return tag, msg

    def npreshape(self, l):
        return np.reshape(l, [1, len(l)])