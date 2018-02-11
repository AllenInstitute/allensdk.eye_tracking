from allensdk.eye_tracking import frame_stream as fs
import sys
import numpy as np
import mock
import pytest

DEFAULT_FRAMES = 101
DEFAULT_CV_FRAMES = 20
# H264 is not available by default on windows
if sys.platform == "win32":
    FOURCC = "FMP4"
else:
    FOURCC = "H264"


def image(shape=(200, 200), value=0):
    return np.ones(shape, dtype=np.uint8)*value*10


@pytest.fixture(scope="module")
def movie(tmpdir_factory):
    frame = image()
    filename = str(tmpdir_factory.mktemp("test").join('movie.avi'))
    ostream = fs.CvOutputStream(filename, frame.shape[::-1], is_color=False,
                                fourcc=FOURCC)
    ostream.open(filename)
    for i in range(DEFAULT_CV_FRAMES):
        ostream.write(image(value=i))
    ostream.close()
    return filename


@pytest.fixture()
def outfile(tmpdir_factory):
    return str(tmpdir_factory.mktemp("test").join("output.avi"))


def test_frame_input_init():
    istream = fs.FrameInputStream("test_path")
    assert(istream.movie_path == "test_path")
    assert(istream.num_frames == 0)
    with pytest.raises(NotImplementedError):
        istream.frame_shape


def test_frame_input_slice_errors():
    mock_cb = mock.MagicMock()
    istream = fs.FrameInputStream("test_path", num_frames=DEFAULT_FRAMES,
                                  process_frame_cb=mock_cb)
    with pytest.raises(NotImplementedError):
        istream._get_frame(20)
    with pytest.raises(NotImplementedError):
        istream._seek_frame(20)
    with pytest.raises(ValueError):
        for x in istream[6:10:0]:
            pass
    with pytest.raises(KeyError):
        istream["invalid"]
    with pytest.raises(IndexError):
        istream[DEFAULT_FRAMES]
    with pytest.raises(IndexError):
        istream[-DEFAULT_FRAMES-1]


@pytest.mark.parametrize("start,stop,step", [
    (5, 10, None),
    (2, 30, 2),
    (30, 2, -2),
    (None, -1, None),
    (3, None, None)
])
def test_frame_input_slice(start, stop, step):
    mock_cb = mock.MagicMock()
    istream = fs.FrameInputStream("test_path", num_frames=DEFAULT_FRAMES,
                                  process_frame_cb=mock_cb)
    with mock.patch.object(istream, "_get_frame", new=lambda a: a):
        count = 0
        for x in istream:
            count += 1
        assert count == DEFAULT_FRAMES
        assert mock_cb.call_count == DEFAULT_FRAMES
        with mock.patch.object(istream, "_seek_frame", new=lambda b: b):
            mock_cb.reset_mock()
            x = istream[5]
            mock_cb.assert_called_once_with(5)
            mock_cb.reset_mock()
            x = istream[-20]
            mock_cb.assert_called_once_with(DEFAULT_FRAMES-20)
        with mock.patch.object(istream, "_seek_frame", new=lambda b: b):
            mock_cb.reset_mock()
            for x in istream[start:stop:step]:
                pass
            rstop = stop if stop is not None else DEFAULT_FRAMES
            rstart = start if start is not None else 0
            rstep = step if step is not None else 1
            expected = [mock.call(x) for x in range(rstart, rstop, rstep)]
            mock_cb.assert_has_calls(expected)


def test_frame_input_context_manager():
    with mock.patch.object(fs.traceback, "print_tb") as mock_tb:
        with pytest.raises(OSError):
            with fs.FrameInputStream("test_path") as istream:
                assert(istream.movie_path == "test_path")
                raise OSError()
            mock_tb.assert_called_once()
    with fs.FrameInputStream("test_path") as istream:
        istream._num_frames = 10
        istream.frames_read = 10


def test_frame_input_close():
    with mock.patch.object(fs.logging, "debug") as mock_debug:
        istream = fs.FrameInputStream("test_path")
        istream.close()
        istream = fs.FrameInputStream("test_path", num_frames=10)
        istream.close()
        assert(mock_debug.call_count == 2)


def test_cv_input_num_frames(movie):
    istream = fs.CvInputStream(movie)
    assert(istream.num_frames == DEFAULT_CV_FRAMES)
    assert(istream.num_frames == DEFAULT_CV_FRAMES)  # using cached value


def test_cv_input_frame_shape(movie):
    istream = fs.CvInputStream(movie)
    assert(istream.frame_shape == (200, 200))
    assert(istream.frame_shape == (200, 200))  # using cached value


def test_cv_input_open(movie):
    istream = fs.CvInputStream(movie)
    istream.open()
    with pytest.raises(IOError):
        istream.open()
    istream._error()
    assert(istream.cap is None)


def test_cv_input_close(movie):
    istream = fs.CvInputStream(movie)
    istream.close()


def test_cv_input_ioerrors(movie):
    istream = fs.CvInputStream(movie)
    with pytest.raises(IOError):
        istream._seek_frame(10)
    with pytest.raises(IOError):
        istream._get_frame(10)


def test_cv_input_iter(movie):
    mock_cb = mock.MagicMock()
    istream = fs.CvInputStream(movie, process_frame_cb=mock_cb)
    count = 0
    for x in istream:
        count += 1
    assert count == DEFAULT_CV_FRAMES
    assert mock_cb.call_count == DEFAULT_CV_FRAMES
    mock_cb.reset_mock()
    for x in istream[5:10]:
        pass
    for i, x in enumerate(range(5, 10)):
        assert(np.all(np.abs(image(value=x) -
                             mock_cb.mock_calls[i][1][0][:, :, 0]) < 2))
    mock_cb.reset_mock()
    for x in istream[2:18:2]:
        pass
    for i, x in enumerate(range(2, 18, 2)):
        assert(np.all(np.abs(image(value=x) -
                             mock_cb.mock_calls[i][1][0][:, :, 0]) < 2))
    mock_cb.reset_mock()
    for x in istream[18:2:-2]:
        pass
    for i, x in enumerate(range(18, 2, -2)):
        assert(np.all(np.abs(image(value=x) -
                             mock_cb.mock_calls[i][1][0][:, :, 0]) < 2))


def test_frame_output_init():
    ostream = fs.FrameOutputStream(200)
    assert(ostream.frames_processed == 0)
    assert(ostream.block_size == 200)


def test_frame_output_open():
    ostream = fs.FrameOutputStream()
    ostream.frames_processed = 1
    ostream.open("test")
    assert(ostream.frames_processed == 0)
    assert(ostream.movie_path == "test")


def test_frame_output_close():
    with mock.patch.object(fs.FrameOutputStream, "_write_frames") as write:
        ostream = fs.FrameOutputStream()
        ostream.block_frames = [1, 2]
        ostream.close()
        write.assert_called_once_with([1, 2])


def test_frame_output_context_manager():
    with mock.patch.object(fs.FrameOutputStream, "close") as mock_close:
        with pytest.raises(OSError):
            with fs.FrameOutputStream() as ostream:
                raise OSError()
        with fs.FrameOutputStream() as ostream:  # noqa: F841
            pass
        mock_close.assert_called_once()


def test_frame_output_write():
    with pytest.raises(NotImplementedError):
        ostream = fs.FrameOutputStream()
        ostream.write(1)
    with mock.patch.object(fs.FrameOutputStream, "_write_frames") as write:
        ostream = fs.FrameOutputStream()
        ostream.write(1)
        write.assert_called_once()
    with mock.patch.object(fs.FrameOutputStream, "_write_frames") as write:
        ostream = fs.FrameOutputStream(block_size=50)
        ostream.write(1)
        ostream.close()
        write.assert_called_once()


def test_cv_output_open(outfile):
    ostream = fs.CvOutputStream(outfile, (200, 200), fourcc=FOURCC)
    ostream.open(outfile)
    assert(ostream.movie_path == outfile)
    with pytest.raises(IOError):
        ostream.open(outfile)


def test_cv_output_context_manager(outfile):
    with pytest.raises(IOError):
        with fs.CvOutputStream(outfile, (200, 200), fourcc=FOURCC) as ostream:
            pass
    with pytest.raises(IOError):
        with fs.CvOutputStream(outfile, (200, 200), fourcc=FOURCC) as ostream:
            ostream.open(outfile)
            ostream.open(outfile)


def test_cv_output_write(outfile):
    ostream = fs.CvOutputStream(outfile, (200, 200), is_color=False,
                                fourcc=FOURCC)
    ostream.write(image())
    ostream.write(image())
    ostream.close()
    check = fs.CvInputStream(outfile)
    assert(check.num_frames == 2)
