import numpy as np


def one_hot(self, indices, depth):
    """Simple implementation of tensorflow.one_hot function.
        Ref: https://www.tensorflow.org/api_docs/python/tf/one_hot

        Args:
            indices (np.array)
            depth (int)
        Returns:
            output (np.array), one-hot tensor
    """
    return np.eye(depth, dtype=indices.dtype)[indices]
