def cuda(ts, gpu_id, **kwargs):
    """Apply tensor.cuda() if gpu_id is valid
        Args:
            ts (torch.tensor)
            gpu_id (int) - The destination GPU id. -1 for CPU only.
        Returns:
            ts (torch.tensor)
    """
    assert gpu_id >= -1
    if gpu_id >= 0:
        return ts.cuda(gpu_id, **kwargs)
    return ts
