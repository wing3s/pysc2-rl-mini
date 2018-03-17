import signal
import psutil


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """Recursively kill child processes given pid"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)
