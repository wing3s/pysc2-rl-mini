import os
import time
import argparse
import torch
import torch.multiprocessing as mp
from absl import flags

from envs import GameInterfaceHandler
from model import FullyConv
from optim import SharedAdam
from worker import worker_fn
from monitor import monitor_fn
from summary import writer_fn, Summary
from utils.sysprocess import kill_child_processes

# workaround for pysc2 flags
FLAGS = flags.FLAGS
FLAGS([__file__])

parser = argparse.ArgumentParser(description='A3C')

# model parameters
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1e-5)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-weight', type=float, default=1e-3, metavar='EW',
                    help='weight of entropy loss (default: 1e-3)')
parser.add_argument('--max-grad-norm', type=float, default=10, metavar='MGN',
                    help='max gradient norm to avoid explode (default:10)')
parser.add_argument('--lstm', type=bool, default=True, metavar='LSTM',
                    help='enable LSTM (default: True)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed (default: 123)')

# experiment parameters
parser.add_argument('--map-name', default='FindAndDefeatZerglings', metavar='MAP',
                    help='environment(mini map) to train on (default: FindAndDefeatZerglings)')
parser.add_argument('--job-name', default='default', metavar='JN',
                    help='job name for identification (default: "default")')
parser.add_argument('--num-processes', type=int, default=4, metavar='NP',
                    help='number of training processes to use (default: 4)')
parser.add_argument('--gpu-ids', type=int, default=[-1], nargs='+',
                    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--num-forward-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='MEL',
                    help='max length of an episode (default: 100000)')
parser.add_argument('--max-k-episode', type=int, default=-1, metavar='MNE',
                    help='max number of total episodes to run in thousands (default: infinite)')
parser.add_argument('--reset', action='store_true',
                    help='If set, delete the existing model and start training from scratch')

# system parameters
parser.add_argument('--model-dir', default='output/models', metavar='MD',
                    help='folder to save/load trained models (default: .output/models)')
parser.add_argument('--log-dir', default='output/logs', metavar='LD',
                    help='folder to save logs (default: .output/logs)')
parser.add_argument('--summary-dir', default='output/summaries', metavar='LD',
                    help='folder to save summaries for Tensorboard (default: .output/summaries)')
parser.add_argument('--summary-iters', type=int, default=8, metavar='SI',
                    help='record training summary after this many update iterations (default: 8)')


def init_dirs(args):
    for dir_path in [args.log_dir, args.model_dir, args.summary_dir]:
        sub_dir_path = "{0}/{1}/{2}".format(dir_path, args.map_name, args.job_name)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)


def main(args):
    init_dirs(args)
    os.environ['OMP_NUM_THREADS'] = '1'  # still required for CPU performance
    mp.set_start_method('spawn')  # required to avoid Conv2d froze issue
    summary_queue = mp.Queue()
    game_intf = GameInterfaceHandler()
    # critic
    shared_model = FullyConv(
        game_intf.minimap_channels,
        game_intf.screen_channels,
        game_intf.screen_resolution,
        game_intf.num_action,
        args.lstm)

    # load or reset model file and logs
    counter_f_path = '{0}/{1}/{2}/counter.log'.format(args.log_dir, args.map_name, args.job_name)
    global_episode_counter_val = 0
    if not args.reset:
        try:
            model_f_path = '{0}/{1}/{2}.dat'.format(args.model_dir, args.map_name, args.job_name)
            shared_model.load_state_dict(torch.load(model_f_path))
            with open(counter_f_path, 'r') as counter_f:
                global_episode_counter_val = int(counter_f.readline())
                summary_queue.put(
                    Summary(action='add_text', tag='log',
                            value1='Reuse trained model {0}, from global_counter: {1}'.format(model_f_path, global_episode_counter_val)))
        except FileNotFoundError as e:
            summary_queue.put(
                Summary(action='add_text', tag='log', value1='No model found -- Start from scratch, {0}'.format(str(e))))
    else:
        summary_queue.put(
            Summary(action='add_text', tag='log', value1='Reset -- Start from scratch'))
    with open(counter_f_path, 'w+') as counter_f:
        counter_f.write(str(global_episode_counter_val))
    summary_queue.put(
        Summary(action='add_text', tag='log', value1='Main process pid: {0}'.format(os.getpid())))
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    # multiprocesses, Hogwild! style update
    processes = []

    global_episode_counter = mp.Value('i', global_episode_counter_val)

    # each worker_thread creates its own environment and trains agents
    for rank in range(args.num_processes):
        # only write summaries in one of the workers, since they are identical
        worker_summary_queue = summary_queue if rank == 0 else None
        worker_thread = mp.Process(
            target=worker_fn, args=(rank, args, shared_model, global_episode_counter, worker_summary_queue, optimizer))
        worker_thread.daemon = True
        worker_thread.start()
        processes.append(worker_thread)
        time.sleep(2)

    # start a thread for policy evaluation
    monitor_thread = mp.Process(
        target=monitor_fn, args=(args.num_processes, args, shared_model, global_episode_counter, summary_queue))
    monitor_thread.daemon = True
    monitor_thread.start()
    processes.append(monitor_thread)

    # summary writer thread
    summary_thread = mp.Process(
        target=writer_fn, args=(args, summary_queue))
    summary_thread.daemon = True
    summary_thread.start()
    processes.append(summary_thread)

    # wait for all processes to finish
    try:
        killed_process_count = 0
        for process in processes:
            process.join()
            killed_process_count += 1 if process.exitcode == 1 else 0
            if killed_process_count >= len(processes) - 2:
                # exit if only monitor and writer alive
                raise SystemExit
    except (KeyboardInterrupt, SystemExit):
        for process in processes:
            # without killing child process, process.terminate() will cause orphans
            # ref: https://thebearsenal.blogspot.com/2018/01/creation-of-orphan-process-in-linux.html
            kill_child_processes(process.pid)
            process.terminate()
            process.join()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
