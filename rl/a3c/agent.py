import os
import time
import torch
import torch.multiprocessing as mp

from rl.envs import GameInterfaceHandler
from rl.model import FullyConv
from rl.optim import SharedAdam
from rl.a3c.worker import worker_fn
from rl.a3c.monitor import monitor_fn
from rl.a3c.summary import writer_fn, Summary
from rl.utils.sys_process import kill_child_processes


def main(args):
    mp.set_start_method('spawn')  # required to avoid Conv2d froze issue
    summary_queue = mp.Queue()
    game_intf = GameInterfaceHandler(args.mode)
    # critic
    shared_model = FullyConv(
        game_intf.minimap_channels,
        game_intf.screen_channels,
        game_intf.screen_resolution,
        game_intf.num_action,
        args.lstm)

    # load or reset model file and logs
    counter_f_path = os.path.join(args.log_dir, args.mode, args.map_name, args.job_name, "counter.log")
    init_episode_counter_val = 0
    if not args.reset:
        try:
            model_f_path = os.path.join(args.model_dir, args.mode, args.map_name, args.job_name, "model.dat")
            shared_model.load_state_dict(torch.load(model_f_path))
            with open(counter_f_path, 'r') as counter_f:
                init_episode_counter_val = int(counter_f.readline())
            summary_queue.put(
                Summary(action='add_text', tag='log',
                        value1='Reuse trained model {0}, from global_counter: {1}'.format(model_f_path, init_episode_counter_val)))
        except FileNotFoundError as e:
            summary_queue.put(
                Summary(action='add_text', tag='log', value1='No model found -- Start from scratch, {0}'.format(str(e))))
    else:
        summary_queue.put(
            Summary(action='add_text', tag='log', value1='Reset -- Start from scratch'))
    with open(counter_f_path, 'w+') as counter_f:
        counter_f.write(str(init_episode_counter_val))
    summary_queue.put(
        Summary(action='add_text', tag='log', value1='Main process PID: {0}'.format(os.getpid())))
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    # multiprocesses, Hogwild! style update
    processes = []

    global_episode_counter = mp.Value('i', init_episode_counter_val)

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
        target=writer_fn, args=(args, summary_queue, init_episode_counter_val))
    summary_thread.daemon = True
    summary_thread.start()
    processes.append(summary_thread)

    # wait for all processes to finish
    try:
        killed_process_count = 0
        for process in processes:
            process.join()
            killed_process_count += 1 if process.exitcode == 1 else 0
            if killed_process_count >= args.num_processes:
                # exit if only monitor and writer alive
                raise SystemExit
    except (KeyboardInterrupt, SystemExit):
        for process in processes:
            # without killing child process, process.terminate() will cause orphans
            # ref: https://thebearsenal.blogspot.com/2018/01/creation-of-orphan-process-in-linux.html
            kill_child_processes(process.pid)
            process.terminate()
            process.join()
