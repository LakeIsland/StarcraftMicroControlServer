from deep_sarsa_server import *
from deep_q_server import *
from deep_q_server import *
import sys, os
if __name__ == "__main__":
    s = DeepSARSAServer()
    # try:
    s.waitForClient()
    # except:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     print("Unexpected error:", sys.exc_info()[0])
    #     if(s.mode == 'train'):
    #         s.export(s.max_iterate)
    #try:
    #
    #except:
    #    if(not s.evaluate):
    #

    #s = DQNServer()
    #s.waitForClient()