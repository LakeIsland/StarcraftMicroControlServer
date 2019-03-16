import socket
from rl_agent.deep_sarsa_agent import *
import ast
import datetime

def npreshape(l):
    return np.reshape(l, [1, len(l)])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("소켓 생성완료")

HOST = 'localhost'
port = 9999
s.bind((HOST, port))
s.listen(5)
print("%d 포트로 연결을 기다리는중" % (port))

while True:
    c, addr = s.accept()
    print(addr, "사용자가 접속함")
    break

max_iterate = 1000

action_size = 2
state_size = 4

agent = DeepSarsaAgent(action_size=action_size, state_size=state_size)
print(max_iterate, " Iterate")

episode = 0
evaluate = False


while episode < max_iterate:
    while True:

        recv_msg = c.recv(1024).decode(encoding='UTF-8').split(":")
        #print(recv_msg)
        tag = recv_msg[0]
        msg = ast.literal_eval(recv_msg[1])

        #print(tag, "tag")
        #print(msg, "msg")
        if tag == "finish":
            print("tag FINISHED")
            break
        elif tag == "state":
            action = agent.get_action(npreshape(msg))
            c.send(bytes(str(action) + '\n', encoding='UTF-8'))
        elif tag == "sarsa":
            sarsa = msg
            agent.train_model(npreshape(sarsa[0]), sarsa[1], sarsa[2], npreshape(sarsa[3]), sarsa[4])
            c.send(bytes('train finished\n', encoding='UTF-8'))
        else:
            print("Unclassified tag", tag)


    episode += 1
    print("Episode %d ended." % (episode))
c.close()
now = datetime.datetime.now()
nowDate = now.strftime('%Y_%m_%d_%H_%M')
agent.model.save_weights("../modeldata/deep_sarsa_%s_%d_times_%s.h5" % ("1Vulture_6Firebat", max_iterate, nowDate))