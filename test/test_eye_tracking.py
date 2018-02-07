from allensdk.eye_tracking import eye_tracking as et
from skimage.draw import circle
import numpy as np
import pytest
from mock import patch


def image(shape=(200, 200), cr_radius=10, cr_center=(100, 100),
          pupil_radius=30, pupil_center=(100, 100)):
    im = np.ones(shape, dtype=np.uint8)*128
    r, c = circle(pupil_center[0], pupil_center[1], pupil_radius, shape)
    im[r, c] = 0
    r, c = circle(cr_center[0], cr_center[1], cr_radius, shape)
    im[r, c] = 255
    return im


class InputStream(object):
    def __init__(self, n_frames=5):
        self.n_frames = n_frames

    def __iter__(self):
        for i in range(self.n_frames):
            yield image()

    @property
    def num_frames(self):
        return self.n_frames


class OutputStream(object):
    def __init__(self, shape):
        self.shape = (shape[0], shape[1], 3)
        self.closed = False

    def write(self, array):
        assert(array.shape == self.shape)

    def close(self):
        self.closed = True


def test_invalid_point_type():
    pg = et.PointGenerator(100, 10, 1.5, 1.5, 10, 10)
    values = np.arange(50, dtype=np.uint8)
    with pytest.raises(ValueError):
        pg.threshold_crossing(pg.xs[0], pg.ys[0], values, "blah")


@pytest.mark.parametrize(("threshold_factor,threshold_pixels,point_type,ray,"
                         "raises"), [
    (1.5, 10, "pupil", 0, False),
    (5, 20, "cr", 5, False),
    (3, 40, "pupil", 0, True)
])
def test_threshold_crossing(threshold_factor, threshold_pixels, point_type,
                            ray, raises):
    pg = et.PointGenerator(100, 10, threshold_factor, threshold_factor,
                           threshold_pixels, threshold_pixels)
    values = np.arange(50, dtype=np.uint8)
    if raises:
        with pytest.raises(ValueError):
            pg.threshold_crossing(pg.xs[ray], pg.ys[ray], values, point_type)
    else:
        t = pg.get_threshold(values, pg.threshold_pixels[point_type],
                             pg.threshold_factor[point_type])
        y, x = pg.threshold_crossing(pg.xs[ray], pg.ys[ray], values,
                                     point_type)
        idx = np.where(pg.xs[ray] == x)
        if pg.above_threshold[point_type]:
            assert(idx == np.argmax(values[threshold_pixels:] > t) +
                   threshold_pixels)
        else:
            assert(idx == np.argmax(values[threshold_pixels:] < t) +
                   threshold_pixels)


@pytest.mark.parametrize("image,seed,above", [
    (image(), (100, 100), True),
    (image(), (100, 100), False)
])
def test_get_candidate_points(image, seed, above):
    pg = et.PointGenerator(100, 10, 1, 10)
    pg.get_candidate_points(image, seed, above)


@pytest.mark.parametrize(("im_shape,input_stream,output_stream,"
                          "starburst_params,ransac_params,pupil_bounding_box,"
                          "cr_bounding_box,kwargs"), [
    ((200, 200),
     None,
     None,
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     {}),
    ((200, 200),
     None,
     None,
     None,
     None,
     [20, 40, 20, 40],
     [20, 40, 20, 40],
     {})
])
def test_eye_tracker_init(im_shape, input_stream, output_stream,
                          starburst_params, ransac_params, pupil_bounding_box,
                          cr_bounding_box, kwargs):
    tracker = et.EyeTracker(im_shape, input_stream, output_stream,
                            starburst_params, ransac_params,
                            pupil_bounding_box, cr_bounding_box, **kwargs)
    assert(tracker.im_shape == im_shape)
    if pupil_bounding_box is None:
        test_pupil_bbox = et.default_bounding_box(im_shape)
    else:
        test_pupil_bbox = pupil_bounding_box
    if cr_bounding_box is None:
        test_cr_bbox = et.default_bounding_box(im_shape)
    else:
        test_cr_bbox = cr_bounding_box
    assert(np.all(tracker.pupil_bounding_box == test_pupil_bbox))
    assert(np.all(tracker.cr_bounding_box == test_cr_bbox))
    assert(input_stream == tracker.input_stream)
    if output_stream is None:
        assert(tracker.annotator.output_stream is None)
    else:
        assert(tracker.annotator.output_stream is not None)


@pytest.mark.parametrize(("image,input_stream,output_stream,"
                          "starburst_params,ransac_params,pupil_bounding_box,"
                          "cr_bounding_box,kwargs"), [
    (image(),
     None,
     None,
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     {}),
    (image(),
     None,
     None,
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     {"recolor_cr": False}),
    (image(cr_center=(85, 25), pupil_center=(70, 100)),
     None,
     None,
     {"n_rays": 20, "cr_threshold_factor": 0, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 0, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     {"recolor_cr": False, "adaptive_pupil": False})
])
def test_process_image(image, input_stream, output_stream,
                       starburst_params, ransac_params, pupil_bounding_box,
                       cr_bounding_box, kwargs):
    im_shape = image.shape
    tracker = et.EyeTracker(im_shape, input_stream, output_stream,
                            starburst_params, ransac_params,
                            pupil_bounding_box, cr_bounding_box, **kwargs)
    with patch.object(tracker, "update_last_pupil_color") as mock_update:
        cr, pupil = tracker.process_image(image)
        if not kwargs.get("adaptive_pupil", True):
            assert mock_update.call_count == 0


@pytest.mark.parametrize(("shape,input_stream,output_stream,"
                          "starburst_params,ransac_params,pupil_bounding_box,"
                          "cr_bounding_box,generate_QC_output,kwargs"), [
    ((200, 200),
     InputStream(),
     None,
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     False,
     {}),
    ((200, 200),
     InputStream(),
     OutputStream((200, 200)),
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     True,
     {}),
])
def test_process_stream(shape, input_stream, output_stream, starburst_params,
                        ransac_params, pupil_bounding_box,
                        cr_bounding_box, generate_QC_output, kwargs):
    tracker = et.EyeTracker(shape, input_stream, output_stream,
                            starburst_params, ransac_params,
                            pupil_bounding_box, cr_bounding_box,
                            generate_QC_output, **kwargs)
    pupil, cr = tracker.process_stream(3)
    assert(pupil.shape == (3, 5))
    pupil, cr = tracker.process_stream(update_mean_frame=False)
    assert(pupil.shape == (input_stream.num_frames, 5))


@pytest.mark.parametrize(("shape,input_stream,output_stream,"
                          "starburst_params,ransac_params,pupil_bounding_box,"
                          "cr_bounding_box,generate_QC_output,kwargs"), [
    ((200, 200),
     InputStream(),
     None,
     {"n_rays": 20, "cr_threshold_factor": 1.4, "cr_threshold_pixels": 5,
      "pupil_threshold_factor": 1.4, "pupil_threshold_pixels": 5,
      "index_length": 100},
     {"iterations": 20, "threshold": 1, "minimum_points_for_fit": 10,
      "number_of_close_points": 3},
     None,
     None,
     False,
     {}),
])
def test_mean_frame(shape, input_stream, output_stream, starburst_params,
                    ransac_params, pupil_bounding_box,
                    cr_bounding_box, generate_QC_output, kwargs):
    tracker = et.EyeTracker(shape, input_stream, output_stream,
                            starburst_params, ransac_params,
                            pupil_bounding_box, cr_bounding_box,
                            generate_QC_output, **kwargs)
    assert(tracker.mean_frame.shape == shape)
