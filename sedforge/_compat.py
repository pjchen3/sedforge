import numpy as np


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Compatibility wrapper for NumPy's trapezoidal integration API."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)
