# Reference: openai/baselines/baselines/common/vec_env/subproc_vec_env
# changes: more meaningful variable names, adapt to pysc2 env
import numpy as np
from torch.multiprocessing import Process, Pipe


def worker(worker_conn, mgr_conn, env_fn_wrapper):
    """ Remote worker to receive command and send back state"""
    mgr_conn.close()
    env = env_func_wrapper.x()
    while True:
        cmd, action = worker_conn.recv()
        if cmd == 'step':
            state = env.step([action])[0]
            if done:
                state = env.reset()[0]
            worker_conn.send(state)
        elif cmd == 'reset':
            state = env.reset()[0]
            worker_conn.send(state)
        elif cmd == 'close':
            worker_conn.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)"""
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(Object):
    def __init__(self, env_fns):
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

    def step(self, actions):
        for mgr_conn, action in zip(self.mgr_conns, actions):
            mgr_conn.send(('step', action))
        obs = [mgr_conn.recv() for mgr_conn in self.mgr_conns]
        return obs

    def reset(self):
        for mgr_conn in self.mgr_conns:
            mgr_conn.send(('reset', None))
        obs = [mgr_conn.recv() for mgr_conn in self.mgr_conns]
        return obs

    def close(self):
        if self.closed:
            return
        for mgr_conn in self.mgr_conns:
            mgr_conn.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
