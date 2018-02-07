from allensdk.eye_tracking import ransac
import numpy as np
import pytest


def parameters(a, b):
    return np.array([b, a])


def line_data(a, b):
    d, _ = np.meshgrid(np.arange(1, 51), np.arange(2))
    d = d.astype(float)
    d[1, :] *= b
    d[1, :] += a
    return d.T


def poly_data(a, b, c):
    d, _ = np.meshgrid(np.arange(1, 51), np.arange(2))
    d = d.astype(float)
    d[1, :] = a + b*d[0, :] + c*(d[0, :])**2
    return d.T


def error_function(params, data):
    res = params[1] + params[0]*data[:, 0]
    return (data[:, 1]-res)**2


def fit_function(data):
    params = np.polyfit(data[:, 0], data[:, 1], 1)
    error = np.mean(error_function(params, data))
    return params, error


@pytest.mark.parametrize("offset,slope,data_offset,threshold", [
    (0, 1.5, 0, 1),
    (20, 0.5, 10, 1)
])
def test_check_outliers(offset, slope, data_offset, threshold):
    params = parameters(offset, slope)
    data = line_data(data_offset, slope)
    outlier_inds = np.arange(5, dtype=np.uint8)
    also_ins = ransac.check_outliers(error_function, params, data,
                                     outlier_inds, threshold)
    if (data_offset - offset)**2 > threshold:
        assert(len(also_ins) == 0)
    else:
        assert(len(also_ins) == 5)


def test_partition_candidate_indices():
    data = line_data(1, 1)
    ins, outs = ransac.partition_candidate_indices(data, 25)
    assert(len(ins) == len(outs) == 25)


def test_fit():
    data = line_data(0, 1)
    rf = ransac.RansacFitter()
    model = rf.fit(fit_function, error_function, data, 1.5, 20, 10, 10)
    assert(np.abs(model[0] - 1) < 0.0000001)
    assert(np.abs(model[1] - 0) < 0.0000001)
    with pytest.raises(ValueError):
        rf.fit(fit_function, error_function, data, 1.5, 200, 10, 10)
    data = poly_data(1, 2, 3)
    model = rf.fit(fit_function, error_function, data, 1.5, 20, 10, 10)
    assert(model is None)
