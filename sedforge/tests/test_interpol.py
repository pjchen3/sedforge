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


def test_interpolate_renormalizes_weights_around_missing_grid_nodes():
    axis_values = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    pixelgrid = np.array([
        [[0.0], [2.0]],
        [[4.0], [np.inf]],
    ])

    center = interpol.interpolate(
        np.array([[0.5], [0.5]]),
        axis_values,
        pixelgrid,
    )

    assert center[0, 0] == np.mean([0.0, 2.0, 4.0])


def test_interpolate_ignores_missing_zero_weight_corner_at_exact_node():
    axis_values = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    pixelgrid = np.array([
        [[3.0], [np.inf]],
        [[np.inf], [np.inf]],
    ])

    exact = interpol.interpolate(
        np.array([[0.0], [0.0]]),
        axis_values,
        pixelgrid,
    )

    assert exact[0, 0] == 3.0


def test_interpolate_returns_nan_outside_grid_or_without_support():
    axis_values = [np.array([0.0, 1.0])]
    pixelgrid = np.array([[np.inf], [np.inf]])

    unsupported = interpol.interpolate(
        np.array([[0.5]]),
        axis_values,
        pixelgrid,
    )
    outside = interpol.interpolate(
        np.array([[1.5]]),
        axis_values,
        np.array([[1.0], [2.0]]),
    )

    assert np.isnan(unsupported[0, 0])
    assert np.isnan(outside[0, 0])


def test_interpolate_preserves_multilinear_values_on_complete_grid():
    x_axis = np.array([0.0, 1.0, 2.0])
    y_axis = np.array([-1.0, 1.0])
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    pixelgrid = np.stack(
        [
            4.0 + 3.0 * x_grid - 2.0 * y_grid,
            -1.0 + 0.5 * x_grid + 5.0 * y_grid,
        ],
        axis=-1,
    )
    points = np.array([
        [0.25, 1.75, 2.0],
        [-0.5, 0.25, 1.0],
    ])

    values = interpol.interpolate(points, [x_axis, y_axis], pixelgrid)

    expected = np.vstack([
        4.0 + 3.0 * points[0] - 2.0 * points[1],
        -1.0 + 0.5 * points[0] + 5.0 * points[1],
    ])
    assert np.allclose(values, expected)
