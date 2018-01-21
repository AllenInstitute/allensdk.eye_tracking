from aibs.eye_tracking import frame_stream as fs
import numpy as np
from skimage.draw import circle
import mock
import pytest

DEFAULT_FRAMES = 101
DEFAULT_CV_FRAMES = 10

def frame_gen(num_frames):
    for i in range(num_frames):
        yield np.ones((200,200,3))


def frame_iter(num_frames=None):
    if num_frames is None:
        num_frames = DEFAULT_FRAMES
    mock_iter = mock.MagicMock(
        return_value=frame_gen(num_frames))
    return mock_iter


def image(shape=(200,200), cr_radius=10, cr_center=(100,100),
          pupil_radius=30, pupil_center=(100,100)):
    im = np.ones(shape, dtype=np.uint8)*128
    r, c = circle(pupil_center[0], pupil_center[1], pupil_radius, shape)
    im[r,c] = 0
    r, c = circle(cr_center[0], cr_center[1], cr_radius, shape)
    im[r,c] = 255
    return im


@pytest.fixture()
def movie(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("test").join('movie.avi'))
    frame = image()
    ostream = fs.CvOutputStream(filename, frame.shape[::-1], is_color=False)
    ostream.open(filename)
    for i in range(DEFAULT_CV_FRAMES):
        ostream.write(frame)
    ostream.close()
    return filename


@pytest.fixture()
def outfile(tmpdir_factory):
    return str(tmpdir_factory.mktemp("test").join("output.avi"))


def test_frame_input_init():
    istream = fs.FrameInputStream("test_path")
    istream._read_iter()
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
    with fs.FrameInputStream("test_path") as istream:
        istream._num_frames = 10
        istream.frames_read = 10


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
    patch_path = "aibs.eye_tracking.frame_stream.FrameInputStream._read_iter"
    with mock.patch(patch_path, frame_iter(num_frames)) as mock_iter:
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
    patch_path = "aibs.eye_tracking.frame_stream.FrameInputStream._read_iter"
    with mock.patch(patch_path, frame_iter()):
        istream = fs.FrameInputStream("test_path")
        with mock.patch.object(fs.scipy.misc, "imsave") as mock_imsave:
            istream.create_images("test", "png")
            assert(mock_imsave.call_count == DEFAULT_FRAMES)


def test_cv_input_num_frames(movie):
    istream = fs.CvInputStream(movie)
    assert(istream.num_frames == DEFAULT_CV_FRAMES)


def test_cv_input_frame_shape(movie):
    istream = fs.CvInputStream(movie)
    assert(istream.frame_shape == (200,200))
    assert(istream.frame_shape == (200,200)) # using cached value


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
    istream.open()
    with pytest.raises(IOError):
        istream.close()


def test_cv_input_iter(movie):
    istream = fs.CvInputStream(movie)
    with pytest.raises(IOError):
        r = istream._read_iter()
        next(r)
    count = 0
    for frame in istream:
        assert(frame.shape == (200,200))
        count += 1
    assert(count == DEFAULT_CV_FRAMES)


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


def test_cv_output_open(outfile):
    ostream = fs.CvOutputStream(outfile, (200,200))
    ostream.open(outfile)
    assert(ostream.movie_path  == outfile)
    with pytest.raises(IOError):
        ostream.open(outfile)


def test_cv_output_context_manager(outfile):
    with pytest.raises(IOError):
        with fs.CvOutputStream(outfile, (200,200)) as ostream:
            pass
    with pytest.raises(OSError):
        with fs.CvOutputStream(outfile, (200,200)) as ostream:
            ostream.open(outfile)
            ostream.open(outfile)


def test_cv_output_write(outfile):
    ostream = fs.CvOutputStream(outfile, (200,200), is_color=False)
    ostream.write(image())
    ostream.write(image())
    ostream.close()
    check = fs.CvInputStream(outfile)
    assert(check.num_frames == 2)
