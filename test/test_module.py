from aibs.eye_tracking import __main__
import sys
import mock
import numpy as np
import pytest


def input_stream(source):
    mock_istream = mock.MagicMock()
    mock_istream.num_frames = 2
    mock_istream.frame_shape = (200,200)
    mock_istream.__iter__ = mock.MagicMock(
        return_value=iter([np.zeros((200,200)), np.zeros((200,200))]))
    return mock_istream


@pytest.fixture()
def input_source(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("test").join('input.avi'))
    with open(filename, "w") as f:
        f.write("")
    return str(filename)


@pytest.fixture()
def input_json(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("test").join('input.json'))
    output_dir = str(tmpdir_factory.mktemp("test"))
    in_json = ('{"starburst": { }, "ransac": { }, "eye_params": { },'
               '"qc": {"generate_plots": true, "output_dir": "%s"}, '
               '"annotation": { }, "cr_bounding_box": [], '
               '"pupil_bounding_box": []}') % output_dir
    with open(filename, "w") as f:
        f.write(in_json)
    return str(filename)


def test_main(input_source, input_json):
    mock_ostream = mock.MagicMock()
    __main__.CvInputStream = input_stream
    __main__.CvOutputStream = mock_ostream
    sys.argv = ["aibs.eye_tracking", "--input_json", input_json,
                "--input_source", input_source]
    with mock.patch.object(__main__.plt, "imsave") as mock_imsave:
        with mock.patch.object(__main__.np, "save") as mock_save:
    
            __main__.main()
        assert(mock_imsave.call_count == 3)
        assert(mock_save.call_count == 2)
    __main__.plt.close("all")
