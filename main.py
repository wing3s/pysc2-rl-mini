import sys
import argparse

from absl import flags
import torch.multiprocessing as mp

from envs import GameInterfaceHandler
from model import FullyConv
from optim import SharedAdam
from train import train_fn
from monitor import monitor_fn

# workaround for pysc2 flags
FLAGS = flags.FLAGS
FLAGS([__file__])

parser = argparse.ArgumentParser(description='A3C')

# model parameters
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--lstm', type=bool, default=True, metavar='LSTM',
                    help='enable LSTM (default: True)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')

# experiment parameters
parser.add_argument('--num-processes', type=int, default=4, metavar='NP',
                    help='number of training processes to use (default: 4)')
parser.add_argument('--num-forward-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='max length of an episode (default: 100000)')
parser.add_argument('--map-name', default='FindAndDefeatZerglings', metavar='MAP',
                    help='environment(mini map) to train on (default: FindAndDefeatZerglings)')


def main():
    mp.set_start_method('spawn')
    args = parser.parse_args()

    game_intf = GameInterfaceHandler()
    # critic
    shared_model = FullyConv(
        game_intf.minimap_channels,
        game_intf.screen_channels,
        game_intf.screen_resolution,
        game_intf.num_action,
        args.lstm)
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    # multiprocesses, Hogwild! style update
    processes = []

    global_counter = mp.Value('i', 0)

    # each actor_thread creates its own environment and trains agents
    for rank in range(args.num_processes):
        actor_thread = mp.Process(
            target=train_fn, args=(rank, args, shared_model, global_counter, optimizer))
        actor_thread.daemon = True
        actor_thread.start()
        processes.append(actor_thread)

    # start a thread for policy evaluation
    monitor_thread = mp.Process(
        target=monitor_fn, args=(args.num_processes, args, shared_model, global_counter))
    monitor_thread.daemon = True
    monitor_thread.start()
    processes.append(monitor_thread)

    # wait for all processes to finish
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()

if __name__ == '__main__':
    main()
