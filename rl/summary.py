import os
import datetime
import collections
from tensorboardX import SummaryWriter
from utils.sysprocess import del_dir_contents, get_first_subdir


def writer_fn(args, msg_queue, init_counter_val):
    """subscriber listens to message queue to write to file"""
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir_path = os.path.join(args.summary_dir, args.map_name, args.job_name)

    if init_counter_val > 0 and get_first_subdir(summary_dir_path) is not None:
        # carry over existing stats
        summary_job_dir_path = get_first_subdir(summary_dir_path)
    else:
        # start recording from scratch
        del_dir_contents(summary_dir_path)
        summary_job_dir_path = os.path.join(summary_dir_path, dt_str)

    summary_writer = SummaryWriter(summary_job_dir_path)
    with summary_writer:
        while True:
            summary = msg_queue.get()
            if summary.action in ['add_scalar', 'add_text']:
                action = getattr(summary_writer, summary.action)
                action(summary.tag, summary.value1, summary.global_step)
            elif summary.action == 'add_histogram':
                summary_writer.add_histogram(summary.tag, summary.value1, summary.global_step, bins='auto')
            elif summary.action == 'add_graph':
                summary_writer.add_graph(summary.value1, summary.value2)

Summary = collections.namedtuple(
    'Summary',
    ['action', 'tag', 'value1', 'value2', 'global_step'])
Summary.__new__.__defaults__ = (None,) * len(Summary._fields)
