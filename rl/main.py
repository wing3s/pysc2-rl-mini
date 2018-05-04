import os
import argparse

from a3c import agent


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
parser.add_argument('--mode', default='lite', metavar='M',
                    help='run experiment in full, lite or test mode (default: lite)')
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
    """Set dir path from argparse is relative to project root folder"""
    abs_base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

    for dir_arg in ['log_dir', 'model_dir', 'summary_dir']:
        dir_path = getattr(args, dir_arg)
        if abs_base_path not in dir_path:
            dir_path = os.path.join(abs_base_path, dir_path)
            setattr(args, dir_arg, dir_path)

        sub_dir_path = os.path.join(dir_path, args.mode, args.map_name, args.job_name)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)


def main(args):
    init_dirs(args)
    os.environ['OMP_NUM_THREADS'] = '1'  # still required for CPU performance
    agent.main(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
