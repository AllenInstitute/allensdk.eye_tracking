from aibs.eye_tracking import frame_stream as fs
import numpy as np
import mock
import pytest

DEFAULT_FRAMES = 101
DEFAULT_CV_FRAMES = 20

def frame_gen(num_frames):
    for i in range(num_frames):
        yield np.ones((200,200,3))


def frame_iter(num_frames=None):
    if num_frames is None:
        num_frames = DEFAULT_FRAMES
    mock_iter = mock.MagicMock(
        return_value=frame_gen(num_frames))
    return mock_iter


@pytest.fixture
def mock_cv():
    mock_cv2 = mock.MagicMock()
    mock_cv2.CAP_PROP_FRAME_COUNT = DEFAULT_CV_FRAMES
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 200
    mock_cv2.CAP_PROP_FRAME_WIDTH = 200
    mock_cv2.VideoWriter_fourcc = mock.MagicMock(side_effect=lambda *v: v)
    
    mock_capture = mock.MagicMock()
    mock_capture.get = mock.MagicMock(side_effect=lambda v: v)
    mock_capture.isOpened = mock.MagicMock(return_value=True)
    mock_capture.release = mock.MagicMock()
    mock_capture.read = mock.MagicMock(
        return_value=(True, np.ones((200,200,3))))

    mock_writer = mock.MagicMock()
    mock_writer.release = mock.MagicMock()
    mock_writer.write = mock.MagicMock()

    mock_cv2.VideoCapture = mock.MagicMock(return_value=mock_capture)
    mock_cv2.VideoWriter = mock.MagicMock(return_value=mock_writer)
    return mock_cv2


def test_frame_input_init():
    istream = fs.FrameInputStream("test_path")
    assert(istream.movie_path == "test_path")
    assert(istream.block_size == 1)
    assert(istream.cache_frames == False)
    assert(istream.num_frames is None)
    with pytest.raises(NotImplementedError):
        istream.frame_shape


def test_frame_input_context_manager():
    with mock.patch.object(fs.traceback, "print_tb") as mock_tb:
        with pytest.raises(OSError):
            with fs.FrameInputStream("test_path") as istream:
                assert(istream.movie_path == "test_path")
                raise OSError()
            mock_tb.assert_called_once()


def test_frame_input_close():
    with mock.patch.object(fs.logging, "debug") as mock_debug:
        istream = fs.FrameInputStream("test_path")
        istream.close()
        istream = fs.FrameInputStream("test_path", num_frames=10)
        with pytest.raises(IOError):
            istream.close()
        assert(mock_debug.call_count == 2)


@pytest.mark.parametrize("num_frames,block_size,cache_frames", [
    (None, 1, False),
    (30, None, False),
    (30, 5, True),
])
def test_frame_input_iter(num_frames, block_size, cache_frames):
    if num_frames is None:
        n = DEFAULT_FRAMES
    else:
        n = num_frames
    mock_iter = frame_iter(num_frames)
    fs.FrameInputStream._read_iter = mock_iter
    istream = fs.FrameInputStream("test_path", num_frames=num_frames,
                                  block_size=block_size,
                                  cache_frames=cache_frames)
    for frame in istream:
        assert(frame.shape == (200,200))
    mock_iter.assert_called_once()
    assert(istream.frames_read == n)
    if cache_frames:
        assert(len(istream.frame_cache) == n)
        for frame in istream:
            assert(frame.shape == (200,200))


def test_frame_input_create_images():
    mock_iter = frame_iter()
    fs.FrameInputStream._read_iter = mock_iter
    istream = fs.FrameInputStream("test_path")
    with mock.patch.object(fs.scipy.misc, "imsave") as mock_imsave:
        istream.create_images("test", "png")
        assert(mock_imsave.call_count == DEFAULT_FRAMES)


def test_cv_input_num_frames(mock_cv):
    fs.cv2 = mock_cv
    istream = fs.CvInputStream("test")
    assert(istream.num_frames == DEFAULT_CV_FRAMES)


def test_cv_input_frame_shape(mock_cv):
    fs.cv2 = mock_cv
    istream = fs.CvInputStream("test")
    assert(istream.frame_shape == (200,200))


def test_cv_input_open(mock_cv):
    fs.cv2 = mock_cv
    istream = fs.CvInputStream("test")
    istream.open()
    with pytest.raises(IOError):
        istream.open()
        mock_cv.VideoCapture.assert_called_once_with("test")


def test_cv_input_close(mock_cv):
    fs.cv2 = mock_cv
    istream = fs.CvInputStream("test")
    istream.close()
    istream.open()
    with pytest.raises(IOError):
        istream.close()
    mock_cv.VideoCapture.assert_called_with("test")


def test_cv_input_iter(mock_cv):
    fs.cv2 = mock_cv
    istream = fs.CvInputStream("test")
    with pytest.raises(IOError):
        r = istream._read_iter()
        next(r)
    for frame in istream:
        assert(frame.shape == (200,200))
    assert(mock_cv.VideoCapture().read.call_count == DEFAULT_CV_FRAMES)


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
    with mock.patch.object(fs.FrameOutputStream,"_write_frames") as mock_write:
        ostream = fs.FrameOutputStream()
        ostream.block_frames = [1, 2]
        ostream.close()
        mock_write.assert_called_once_with([1,2])


def test_frame_output_context_manager():
    with mock.patch.object(fs.FrameOutputStream, "close") as mock_close:
        with pytest.raises(OSError):
            with fs.FrameOutputStream() as ostream:
                raise OSError()
        with fs.FrameOutputStream() as ostream:
            pass
        mock_close.assert_called_once()


def test_frame_output_write():
    with pytest.raises(NotImplementedError):
        ostream = fs.FrameOutputStream()
        ostream.write(1)
    with mock.patch.object(fs.FrameOutputStream,"_write_frames") as mock_write:
        ostream = fs.FrameOutputStream()
        ostream.write(1)


def test_cv_output_open(mock_cv):
    fs.cv2 = mock_cv
    ostream = fs.CvOutputStream("test", (200,200))
    ostream.open("test2")
    assert(ostream.movie_path  == "test2")
    with pytest.raises(IOError):
        ostream.open("test3")


def test_cv_output_context_manager(mock_cv):
    fs.cv2 = mock_cv
    with pytest.raises(IOError):
        with fs.CvOutputStream("test", (200,200)) as ostream:
            pass
    with pytest.raises(IOError):
        with fs.CvOutputStream("test", (200,200)) as ostream:
            ostream.open("test")
            raise(IOError)
        mock_cv.VideoWriter().release.assert_called_once()
    with fs.CvOutputStream("test", (200,200)) as ostream:
        ostream.open("test")
    assert(mock_cv.VideoWriter().release.call_count == 2)
    assert(mock_cv.VideoWriter_fourcc.call_count == 3)


def test_cv_output_write(mock_cv):
    fs.cv2 = mock_cv
    ostream = fs.CvOutputStream("test", (200,200))
    ostream.write(1)
    ostream.write(2)
    assert(mock_cv.VideoWriter().write.call_count == 2)
