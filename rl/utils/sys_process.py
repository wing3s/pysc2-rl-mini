import os
import signal
import psutil
import shutil


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """Recursively kill child processes given pid"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


def del_dir_contents(dir_path):
    """Delete all contents given folder path"""
    assert os.path.isdir(dir_path)
    for f in os.listdir(dir_path):
        f_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(f_path):
                os.unlink(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
        except Exception as e:
            print(str(e))


def get_first_subdir(dir_path):
    """Get first subfoler path for given foler path"""
    assert os.path.isdir(dir_path)
    for f in os.listdir(dir_path):
        f_path = os.path.join(dir_path, f)
        if os.path.isdir(f_path):
            return f_path
    return None
