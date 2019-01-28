import socket
import pickle
import numpy as np
import ast
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("소켓 생성완료")

HOST = 'localhost'
port = 1234
s.bind((HOST, port))
s.listen(5)
print("%d 포트로 연결을 기다리는중" % (port))
while True:
    c, addr = s.accept()
    print(addr, "사용자가 접속함")
    break

while True:
    data = c.recv(1024).decode(encoding='UTF-8').replace("[", "").replace("]", "")
    a = np.fromstring(data, sep=',')
    print(data)
    print(a)
    msg = 'dddd\n'
    c.send(bytes(msg,'UTF-8'))
    print('msg sended')
    data2 = c.recv(1024).decode(encoding='UTF-8')
    aaa = ast.literal_eval(data2)
    print(aaa)