import numpy as np

from sedforge import interpol


def test_interpolate_handles_single_value_axis_and_lower_edge():
    axis_values = [np.array([1.0]), np.array([0.0, 1.0])]
    pixelgrid = np.array([[[10.0], [20.0]]])

    lower_edge = interpol.interpolate(
        np.array([[1.0], [0.0]]),
        axis_values,
        pixelgrid,
    )
    upper_edge = interpol.interpolate(
        np.array([[1.0], [1.0]]),
        axis_values,
        pixelgrid,
    )

    assert np.allclose(lower_edge[:, 0], [10.0])
    assert np.allclose(upper_edge[:, 0], [20.0])
