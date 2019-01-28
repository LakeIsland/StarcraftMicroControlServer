import socket
import pickle
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
    # c.send(msg)
    l = pickle.loads(c.recv(1024))
    print(" ".join([str(x) for x in l]))
    # try:
    #     l = pickle.loads(c.recv(1024))
    #     print(" ".join([str(x) for x in l]))
    # except:
    #     pass