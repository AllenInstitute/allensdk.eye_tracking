import matplotlib
from aibs.eye_tracking import plotting
import mock
import numpy as np
import pytest


class WriteEvaluator(object):
    def __init__(self, shape):
        self.shape = (shape[0], shape[1], 3)
        self.closed = False

    def write(self, array):
        assert(array.shape == self.shape)

    def close(self):
        self.closed = True


def frame(height, width):
    return np.zeros((height, width), dtype=np.uint8)


@pytest.mark.parametrize("frame,pupil_params,cr_params", [
    (frame(100,100), (40, 50, 45, 10, 8), (30, 60, 0, 5, 4))
])
def test_annotate_frame(frame, pupil_params, cr_params):
    ostream = WriteEvaluator(frame.shape)
    annotator = plotting.Annotator(ostream)
    annotator.annotate_frame(frame, pupil_params, cr_params)
    annotator.close()
    assert(ostream.closed)


@pytest.mark.parametrize("pupil_params,cr_params,output_folder", [
    ((40, 50, 45, 10, 8), (30, 60, 0, 5, 4), None),
    ((40, 50, 45, 10, 8), (30, 60, 0, 5, 4), None),
])
def test_plot_summary(pupil_params,cr_params,output_folder):
    with mock.patch.object(plotting.plt, "show") as mock_show:
        plotting.plot_summary(pupil_params, cr_params, output_folder, False)
        plotting.plot_summary(pupil_params, cr_params, output_folder, True)
        mock_show.assert_called_once

    plotting.plt.close("all")