# Reference: openai/baselines/baselines/common/vec_env/subproc_vec_env
import numpy as np
from torch.multiprocessing import Process, Pipe


def worker(worker_conn, mgr_conn, env_fn_wrapper):
    """ Remote worker to receive command and send back state"""
    mgr_conn.close()
    env = env_func_wrapper.x()
    while True:
        cmd, data = worker_conn.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_conn.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_conn.send(ob)
        elif cmd == 'close':
            worker_conn.close()
            break
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.mgr_conns, self.worker_conns = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker,
                args=(worker_conn, mgr_conn, CloudpickleWrapper(env_fn)))
            for (worker_conn, mgr_conn, env_fn) in zip(self.worker_conns, self.mgr_conns, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start
        for worker_conn in self.worker_conns:
            worker_conn.close()

        super(SubprocVecEnv, self).__init__()
