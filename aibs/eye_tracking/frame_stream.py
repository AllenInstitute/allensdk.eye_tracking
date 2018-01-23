import logging
import os
import scipy.misc
import traceback
import cv2


class FrameInputStream(object):
    def __init__(self, movie_path, num_frames=None, block_size=1,
                 cache_frames=False, process_frame_cb=None):
        self.movie_path = movie_path
        self._num_frames = num_frames
        self.block_size = block_size
        self.cache_frames = cache_frames
        if process_frame_cb:
            self.process_frame_cb = process_frame_cb
        else:
            self.process_frame_cb = lambda f: f[:, :, 0].copy()
        self.frames_read = 0
        self.frame_cache = []

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

        if self.num_frames is not None and self.frames_read != self.num_frames:
            raise IOError("read incorrect number of frames: %d vs %d",
                          self.frames_read, self.num_frames)

    def _error(self):
        pass

    def _process_frame(self, frame):
        return self.process_frame_cb(frame)

    def _read_iter(self):
        pass

    def __enter__(self):
        return self

    def __iter__(self):
        # if we're caching frames and the cache exists, return it
        if self.cache_frames and self.frame_cache:
            cache_len = len(self.frame_cache)
            n = self.num_frames if self.num_frames is not None else cache_len
            for i in range(n):
                yield self.frame_cache[i]
        else:
            self.open()

            self.frame_cache = []

            for frame in self._read_iter():
                self.frame_cache.append(self._process_frame(frame))
                self.frames_read += 1

                if (self.frames_read % 100) == 0:
                    logging.debug("Read frames %d", self.frames_read)

                if self.block_size is None:
                    continue
                if self.block_size == 1:
                    yield self.frame_cache[-1]
                elif (self.frames_read % self.block_size) == 0:
                    for i in range(-self.block_size, 0):
                        yield self.frame_cache[i]

                if not self.cache_frames:
                    self.frame_cache = []

            self.close()

            for frame in self.frame_cache:
                yield frame

            if not self.cache_frames:
                self.frame_cache = []

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value:
            traceback.print_tb(tb)
            self._error()
            raise exc_value

    def create_images(self, output_directory, image_type):
        for i, frame in enumerate(self):
            file_name = os.path.join(output_directory,
                                     "input_frame-%06d." % i + image_type)
            scipy.misc.imsave(file_name, frame)


class CvInputStream(FrameInputStream):
    def __init__(self, movie_path, num_frames=None, block_size=1,
                 cache_frames=False):
        super(CvInputStream, self).__init__(movie_path=movie_path,
                                            num_frames=num_frames,
                                            block_size=block_size,
                                            cache_frames=cache_frames)
        self.cap = None
        self._frame_shape = None

    @property
    def num_frames(self):
        if self._num_frames is None:
            self.load_capture_properties()
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

    def _read_iter(self):
        if self.cap is None:
            raise IOError("capture is not open")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            yield frame

            if self.frames_read == self.num_frames:
                break

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
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
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
