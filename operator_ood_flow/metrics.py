import numpy as np


def relative_velocity_error(y_pred, y_true):
    """Relative L^2(S^1) error on a uniform periodic grid."""
    return np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
