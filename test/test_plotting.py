from allensdk.eye_tracking import plotting
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


def test_get_rgb_frame():
    img = frame(100,100)
    with pytest.raises(ValueError):
        plotting.get_rgb_frame(np.dstack((img, img)))
    f = plotting.get_rgb_frame(img)
    assert f.shape == (100,100,3)
    f = plotting.get_rgb_frame(np.dstack((img, img, img)))
    assert f.shape == (100,100,3)


@pytest.mark.parametrize("frame,pupil_params,cr_params", [
    (frame(100, 100),
     np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4))),
    (frame(100, 100),
     np.array((np.nan, np.nan, np.nan, np.nan, np.nan)),
     np.array((30, 60, 0, 5, 4))),
    (frame(100, 100),
     np.array((40, 50, 45, 10, 8)),
     np.array((np.nan, np.nan, np.nan, np.nan, np.nan))),
])
def test_annotate_frame(frame, pupil_params, cr_params):
    ostream = WriteEvaluator(frame.shape)
    annotator = plotting.Annotator(ostream)
    annotator.annotate_frame(frame, pupil_params, cr_params)
    annotator.close()
    assert(ostream.closed)

    annotator = plotting.Annotator()
    with mock.patch.object(annotator, "update_rc",
                           mock.MagicMock(return_value=False)):
        with mock.patch("allensdk.eye_tracking.plotting."
                        "color_by_points") as mock_color:
            annotator.annotate_frame(frame, pupil_params, cr_params)
            assert mock_color.call_count == 0
    annotator.annotate_frame(frame, pupil_params, cr_params)
    annotator.close()


@pytest.mark.parametrize("frame,pupil_params,cr_params", [
    (frame(100, 100),
     np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4))),
    (frame(100, 100),
     np.array((np.nan, np.nan, np.nan, np.nan, np.nan)),
     np.array((30, 60, 0, 5, 4))),
])
def test_compute_density(frame, pupil_params, cr_params):
    ostream = WriteEvaluator(frame.shape)
    annotator = plotting.Annotator(ostream)
    assert(annotator._r["pupil"] is None)
    assert(annotator._c["pupil"] is None)
    assert(annotator._r["cr"] is None)
    assert(annotator._c["cr"] is None)
    assert(annotator.densities["pupil"] is None)
    assert(annotator.densities["cr"] is None)
    annotator.compute_density(frame, pupil_params, cr_params)
    if np.any(np.isnan(pupil_params)):
        assert(np.all(annotator.densities["pupil"] == 0))
    else:
        assert(annotator.densities["pupil"].shape == frame.shape)


@pytest.mark.parametrize("frame,pupil_params,cr_params", [
    (frame(100, 100),
     np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4))),
    (frame(100, 100),
     np.array((np.nan, np.nan, np.nan, np.nan, np.nan)),
     np.array((30, 60, 0, 5, 4))),
])
def test_annotate_with_cumulative(frame, pupil_params, cr_params):
    ostream = WriteEvaluator(frame.shape)
    annotator = plotting.Annotator(ostream)
    annotator.compute_density(frame, pupil_params, cr_params)

    with mock.patch.object(plotting.plt, "imsave") as mock_imsave:
        res = annotator.annotate_with_cumulative_pupil(frame, "pupil.png")
        mock_imsave.assert_called_with("pupil.png", mock.ANY)
        assert(res.shape == (frame.shape[0], frame.shape[1], 3))
        res = annotator.annotate_with_cumulative_cr(frame, "cr.png")
        mock_imsave.assert_called_with("cr.png", mock.ANY)
        assert(res.shape == (frame.shape[0], frame.shape[1], 3))

    with mock.patch("allensdk.eye_tracking.plotting."
                    "color_by_mask") as mock_color:
        with mock.patch.object(plotting.plt, "imsave") as mock_imsave:
            res = plotting.annotate_with_cumulative(frame, None)
            assert(mock_color.call_count == 0)
            assert(mock_imsave.call_count == 0)


def test_annotate_with_box():
    img = frame(100,100)
    bbox = np.array((20, 50, 40, 80), dtype="int")
    with mock.patch.object(plotting.plt, "imsave") as mock_imsave:
        res = plotting.annotate_with_box(img, bbox)
        assert(res.shape == (100,100,3))
        assert(mock_imsave.call_count == 0)
        res = plotting.annotate_with_box(img, bbox, filename="test")
        assert(res.shape == (100,100,3))
        mock_imsave.assert_called_once_with("test", res)


@pytest.mark.parametrize("frame,pupil_params,cr_params,output_dir", [
    (frame(100, 100),
     np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4)),
     None),
    (frame(100, 100),
     np.array((np.nan, np.nan, np.nan, np.nan, np.nan)),
     np.array((30, 60, 0, 5, 4)),
     "test"),
])
def test_plot_cumulative(frame, pupil_params, cr_params, output_dir):
    ostream = WriteEvaluator(frame.shape)
    annotator = plotting.Annotator(ostream)
    annotator.compute_density(frame, pupil_params, cr_params)
    with mock.patch.object(plotting.plt, "show") as mock_show:
        with mock.patch.object(plotting.plt.Figure, "savefig") as mock_savefig:
            plotting.plot_cumulative(annotator.densities["pupil"],
                                     annotator.densities["cr"],
                                     output_dir=output_dir,
                                     show=False)
            plotting.plot_cumulative(annotator.densities["pupil"],
                                     annotator.densities["cr"],
                                     output_dir=output_dir,
                                     show=True)
            mock_show.assert_called_once()
            if output_dir is not None:
                assert(mock_savefig.call_count == 4)
            else:
                assert(mock_savefig.call_count == 0)
    plotting.plt.close("all")


@pytest.mark.parametrize("pupil_params,cr_params,output_folder", [
    (np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4)),
     None),
    (np.array((40, 50, 45, 10, 8)),
     np.array((30, 60, 0, 5, 4)),
     "test"),
])
def test_plot_summary(pupil_params, cr_params, output_folder):
    with mock.patch.object(plotting.plt, "show") as mock_show:
        with mock.patch.object(plotting.plt.Figure, "savefig") as mock_savefig:
            plotting.plot_summary(pupil_params, cr_params,
                                  output_folder, False)
            plotting.plot_summary(pupil_params, cr_params, output_folder, True)
            mock_show.assert_called_once()
            if output_folder is not None:
                assert(mock_savefig.call_count == 12)
            else:
                assert(mock_savefig.call_count == 0)
    plotting.plt.close("all")


def test_plots_no_title():
    data = np.arange(50)
    img = frame(100, 100)
    with mock.patch.object(plotting.plt.Axes, "set_title") as mock_set:
        plotting.plot_timeseries(data, None)
        plotting.plot_density(img)
    assert mock_set.call_count == 0
