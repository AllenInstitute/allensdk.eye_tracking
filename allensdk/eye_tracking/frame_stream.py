import logging
import traceback
import cv2


class FrameInputStream(object):
    def __init__(self, movie_path, num_frames=0, block_size=1,
                 cache_frames=False, process_frame_cb=None):
        self.movie_path = movie_path
        self._num_frames = num_frames
        self._start = 0
        self._stop = num_frames
        self._step = 1
        self._i = self._start - self._step
        self._last_i = 0
        self.block_size = block_size
        self.cache_frames = cache_frames
        if process_frame_cb:
            self.process_frame_cb = process_frame_cb
        else:
            self.process_frame_cb = lambda f: f[:, :, 0].copy()
        self.frames_read = 0
        self.frame_cache = []

    def next(self):
        return self.__next__()

    def __getitem__(self, key):
        if isinstance(key, int) and key >= 0:
            self._start = key
            self._stop = key + 1
            self._step = 1
            return list(self)[0]  # force iteration and closing
        elif isinstance(key, slice):
            if key.step == 0:
                raise ValueError("slice step cannot be 0")
            self._start = key.start if key.start is not None else 0
            if key.stop is None:
                self._stop = self.num_frames
            elif key.stop < 0:
                self._stop = max(self.num_frames + key.stop, -1)
            else:
                self._stop = min(self.num_frames, key.stop)
            self._step = key.step if key.step is not None else 1
            return self
        else:
            raise KeyError("Key must be non-negative integer or slice, not {}"
                           .format(key))

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def frame_shape(self):
        raise NotImplementedError(("frame_shape must be implemented in a "
                                   "subclass"))

    def open(self):
        self.frames_read = 0

    def close(self):
        logging.debug("Read total frames %d", self.frames_read)

    def _error(self):
        pass

    def _seek_frame(self, i):
        raise NotImplementedError(("_seek_frame must be implemented in a "
                                   "subclass"))

    def _get_frame(self, i):
        raise NotImplementedError(("_get_frame must be implemented in a "
                                   "subclass"))

    def get_frame(self, i):
        if abs(i - self._last_i) > 1:
            self._seek_frame(i)
        self._last_i = self._i
        self._i = i
        self.frames_read += 1
        if self.frames_read % 100 == 0:
            logging.debug("Read frames %d", self.frames_read)
        return self.process_frame_cb(self._get_frame(self._i))

    def __enter__(self):
        return self

    def __iter__(self):
        self._last_i = 0
        self._i = self._start - self._step
        self.open()
        logging.debug("Iterating over %s from %d to %d by step %d" %
                      (self.movie_path, self._start, self._stop, self._step))
        return self

    def __next__(self):
        self._i = self._i + self._step
        if (self._step < 0 and self._i <= self._stop) or \
           (self._step > 0 and self._i >= self._stop):
            self.close()
            raise StopIteration()
        else:
            return self.get_frame(self._i)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value:
            traceback.print_tb(tb)
            self._error()
            raise exc_value


class CvInputStream(FrameInputStream):
    def __init__(self, movie_path, num_frames=None, block_size=1,
                 cache_frames=False, process_frame_cb=None):
        super(CvInputStream, self).__init__(movie_path=movie_path,
                                            num_frames=num_frames,
                                            block_size=block_size,
                                            cache_frames=cache_frames,
                                            process_frame_cb=process_frame_cb)
        self.cap = None
        self._frame_shape = None
        self._stop = self.num_frames

    @property
    def num_frames(self):
        if self._num_frames is None:
            self.load_capture_properties()
            self._stop = self._num_frames
        return self._num_frames

    @property
    def frame_shape(self):
        if self._frame_shape is None:
            self.load_capture_properties()
        return self._frame_shape

    def load_capture_properties(self):
        close_after = False
        if self.cap is None:
            close_after = True
            self.cap = cv2.VideoCapture(self.movie_path)

        self._num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_shape = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        if close_after:
            self.cap.release()
            self.cap = None

    def open(self):
        if self.cap:
            raise IOError("capture is open already")

        super(CvInputStream, self).open()

        self.cap = cv2.VideoCapture(self.movie_path)
        logging.debug("opened capture")

    def close(self):
        if self.cap is None:
            return

        self.cap.release()
        self.cap = None

        super(CvInputStream, self).close()

    def _seek_frame(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def _get_frame(self, i):
        if self.cap is None:
            raise IOError("capture is not open")
        ret, frame = self.cap.read()
        return frame

    def _error(self):
        self.cap.release()
        self.cap = None


class FrameOutputStream(object):
    def __init__(self, block_size=1):
        self.frames_processed = 0
        self.block_frames = []
        self.block_size = block_size

    def open(self, movie_path):
        self.frames_processed = 0
        self.block_frames = []
        self.movie_path = movie_path

    def _write_frames(self, frames):
        raise NotImplementedError()

    def write(self, frame):
        self.block_frames.append(frame)

        if len(self.block_frames) == self.block_size:
            self._write_frames(self.block_frames)
            self.frames_processed += len(self.block_frames)
            self.block_frames = []

    def close(self):
        if self.block_frames:
            self._write_frames(self.block_frames)
            self.frames_processed += len(self.block_frames)
            self.block_frames = []

        logging.debug("wrote %d frames", self.frames_processed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value:
            raise exc_value
        self.close()


class CvOutputStream(FrameOutputStream):
    def __init__(self, movie_path, frame_shape, frame_rate=30.0,
                 fourcc="H264", is_color=True, block_size=1):
        super(CvOutputStream, self).__init__(block_size)

        self.frame_shape = frame_shape
        self.movie_path = movie_path
        self.fourcc = cv2.VideoWriter_fourcc(*str(fourcc))
        self.frame_rate = frame_rate
        self.is_color = is_color
        self.writer = None

    def open(self, movie_path):
        super(CvOutputStream, self).open(movie_path)

        if self.writer:
            raise IOError("video writer is open already")

        self.writer = cv2.VideoWriter(movie_path, self.fourcc,
                                      self.frame_rate, self.frame_shape,
                                      self.is_color)
        logging.debug("opened video writer")

    def _write_frames(self, frames):
        if self.writer is None:
            self.open(self.movie_path)

        for frame in frames:
            self.writer.write(frame)

    def close(self):
        super(CvOutputStream, self).close()
        if self.writer is None:
            raise IOError("video writer is closed")

        self.writer.release()

        logging.debug("closed video writer")
        self.writer = None

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value:
            self.writer.release()
            raise exc_value
        self.close()
