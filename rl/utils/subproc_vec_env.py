# Reference: openai/baselines/baselines/common/vec_env/subproc_vec_env
import numpy as np
from torch.multiprocessing import Process, Pipe


def worker(conn_worker, conn_parent, env_fn_wrapper):
    """ Remote worker to receive command and send back state"""
    conn_parent.close()
    env = env_func_wrapper.x()
    while True:
        cmd, data = conn_worker.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            conn_worker.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            conn_worker.send(ob)
        elif cmd == 'close':
            conn_worker.close()
            break
        else:
            raise NotImplementedError
