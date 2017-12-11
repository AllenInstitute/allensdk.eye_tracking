import logging
from scipy.signal import medfilt2d
import numpy as np

from .fit_ellipse import EllipseFitter, not_on_ellipse
from .utils import generate_ray_indices, get_ray_values
from .feature_extraction import (get_circle_mask, max_image_at_value,
                                 max_convolution_positions)
from .plotting import Annotator, ellipse_points

SMOOTHING_KERNEL_SIZE = 3
DEFAULT_MIN_PUPIL_VALUE = 0
DEFAULT_MAX_PUPIL_VALUE = 30
DEFAULT_CR_RECOLOR_SCALE_FACTOR = 2.0
DEFAULT_CR_MASK_RADIUS = 10
DEFAULT_PUPIL_MASK_RADIUS = 40


class PointGenerator(object):
    """Class to find candidate points for ellipse fitting.

    Candidates points are found by drawing rays from a seed point and
    checking for the first threshold crossing of each ray.

    Parameters
    ----------
    index_length : int
        Initial default length for ray indices.
    n_rays : int
        The number of rays to check.
    threshold_factor : float
        Multiplicative factor for threshold.
    threshold_pixels : int
        Number of pixels (from beginning of ray) to use to determine
        threshold.
    """
    def __init__(self, index_length, n_rays, threshold_factor,
                 threshold_pixels):
        self.xs, self.ys = generate_ray_indices(index_length, n_rays)
        self.threshold_pixels = threshold_pixels
        self.threshold_factor = threshold_factor

    def get_candidate_points(self, image, seed_point, above_threshold=True,
                             filter_function=None, filter_args=()):
        """Get candidate points for ellipse fitting.

        Parameters
        ----------
        image : np.ndarray
            Image to check for threshold crossings.
        seed_point : tuple
            (y, x) center point for ray burst.
        above_threshold : bool
            Whether looking for transitions above or below a threshold.

        Returns
        -------
        list
            List of (y, x) candidate points.
        """
        xs = self.xs + seed_point[1]
        ys = self.ys + seed_point[0]
        ray_values = get_ray_values(xs, ys, image)
        filtered_out = 0
        threshold_not_crossed = 0
        candidate_points = []
        for i, values in enumerate(ray_values):
            try:
                point = self.threshold_crossing(xs[i], ys[i], values,
                                                above_threshold)
                if filter_function is not None:
                    if filter_function(point, *filter_args):
                        candidate_points.append(point)
                    else:
                        filtered_out += 1
                else:
                    candidate_points.append(point)
            except ValueError:
                threshold_not_crossed += 1
        if threshold_not_crossed or filtered_out:
            logging.debug(("%s candidate points returned, %s filtered out, %s "
                           "not generated because threshold not crossed"),
                          len(candidate_points), filtered_out,
                          threshold_not_crossed)
        return candidate_points

    def threshold_crossing(self, xs, ys, values, above_threshold=True):
        """Check a ray for where it crosses a threshold.

        The threshold is calculated using `get_threshold`.

        Parameters
        ----------
        xs : np.ndarray
            X indices of ray.
        ys : np.ndarray
            Y indices of ray.
        values : np.ndarray
            Image values along ray.
        above_threshold : bool
            Whether to look for transitions above or below a threshold.

        Returns
        -------
        int
            Y index of threshold crossing.
        int
            X index of threshold crossing.

        Raises
        ------
        ValueError
            If no threshold crossing found.
        """
        threshold = self.get_threshold(values)
        if above_threshold:
            comparison = values[self.threshold_pixels:] > threshold
        else:
            comparison = values[self.threshold_pixels:] < threshold
        sub_index = np.argmax(comparison)
        if comparison[sub_index]:
            index = self.threshold_pixels + sub_index
            return ys[index], xs[index]
        else:
            raise ValueError("No value in array crosses: {}".format(threshold))

    def get_threshold(self, ray_values):
        """Calculate the threshold from the ray values.

        The threshold is determined from `threshold_factor` times the
        mean of the first `threshold_pixels` values.

        Parameters
        ----------
        ray_values : np.ndarray
            Values of the ray.
        
        Returns
        -------
        float
            Threshold to set for candidate point.
        """
        sub_ray = ray_values[:self.threshold_pixels]

        return self.threshold_factor*np.mean(sub_ray)


class EyeTracker(object):
    """Mouse Eye-Tracker.
    
    Parameters
    ----------
    im_shape : tuple
        (height, width) of images.
    input_stream : generator
        Generator that yields numpy.ndarray frames to analyze.
    output_stream : stream
        Stream that accepts numpuy.ndarrays in the write method. None if
        not outputting annotations.
    starburst_params : dict
        Dictionary of keyword arguments for `PointGenerator`.
    ransac_params : dict
        Dictionary of keyword arguments for `EllipseFitter`.
    pupil_bounding_box : numpy.ndarray
        [xmin xmax ymin ymax] bounding box for pupil seed point search.
    cr_bounding_box : numpy.ndarray
        [xmin xmax ymin ymax] bounding box for cr seed point search.
    generate_QC_output : bool
        Flag to compute extra QC data on frames.
    **kwargs
        pupil_min_value : int
        pupil_max_value : int
        cr_mask_radius : int
        pupil_mask_radius : int
        cr_recolor_scale_factor : float
        recolor_cr : bool
    """
    def __init__(self, im_shape, input_stream, output_stream, starburst_params,
                 ransac_params, pupil_bounding_box=None, cr_bounding_box=None,
                 generate_QC_output=False, **kwargs):
        self.point_generator = PointGenerator(**starburst_params)
        self.ellipse_fitter = EllipseFitter(**ransac_params)
        self.annotator = Annotator(output_stream)
        self.im_shape = im_shape
        self.input_stream = input_stream
        if pupil_bounding_box is None or len(pupil_bounding_box) == 0:
            pupil_bounding_box = default_bounding_box(im_shape)
        if cr_bounding_box is None or len(cr_bounding_box) == 0:
            cr_bounding_box = default_bounding_box(im_shape)
        self.pupil_bounding_box = pupil_bounding_box
        self.cr_bounding_box = cr_bounding_box
        self.pupil_parameters = []
        self.cr_parameters = []
        self.generate_QC_output = generate_QC_output
        self.current_image = None
        self.blurred_image = None
        self.cr_filled_image = None
        self.pupil_max_image = None
        self._mean_frame = None
        self._init_kwargs(**kwargs)
        self.frame_index = 0

    def _init_kwargs(self, **kwargs):
        self.min_pupil_value = kwargs.get("min_pupil_value",
                                          DEFAULT_MIN_PUPIL_VALUE)
        self.max_pupil_value = kwargs.get("max_pupil_value",
                                          DEFAULT_MAX_PUPIL_VALUE)
        self.last_pupil_color = self.min_pupil_value
        self.cr_recolor_scale_factor = kwargs.get(
            "cr_recolor_scale_factor", DEFAULT_CR_RECOLOR_SCALE_FACTOR)
        self.recolor_cr = kwargs.get("recolor_cr", True)
        self.cr_mask = get_circle_mask(
            kwargs.get("cr_mask_radius", DEFAULT_CR_MASK_RADIUS))
        self.pupil_mask = get_circle_mask(
            kwargs.get("pupil_mask_radius", DEFAULT_PUPIL_MASK_RADIUS))

    @property
    def mean_frame(self):
        if self._mean_frame is None:
            mean_frame = np.zeros(self.im_shape, dtype=np.float64)
            frame_count = 0
            for frame in self.input_stream:
                mean_frame += frame
                frame_count += 1
            self._mean_frame = (mean_frame / frame_count).astype(np.uint8)
        return self._mean_frame

    def find_corneal_reflection(self):
        """Estimate the position of the corneal reflection.

        Returns
        -------
        numpy.ndarray
            [x, y, r, a, b] ellipse parameters.
        """
        seed_point = max_convolution_positions(self.blurred_image,
                                               self.cr_mask,
                                               self.cr_bounding_box)
        candidate_points = self.point_generator.get_candidate_points(
            self.blurred_image, seed_point, False)
        return self.ellipse_fitter.fit(candidate_points)

    def setup_pupil_finder(self, cr_parameters):
        """Initialize image and ransac filter for pupil fitting.

        If recoloring the corneal_reflection, color it in and provide a
        filter to exclude points that fall on the colored-in ellipse
        from fitting.

        Parameters
        ----------
        cr_parameters : numpy.ndarray
            [x, y, r, a, b] ellipse parameters for corneal reflection.

        Returns
        -------
        numpy.ndarray
            Image for pupil fitting. Has corneal reflection filled in if
            `recolor_cr` is set.
        callable
            Function to indicate if points fall on the recolored ellipse
            or None if not recoloring.
        numpy.ndarray
            Ellipse parameters for recolor ellipse shape, which are
            `cr_parameters` with the axes scaled by
            `cr_recolor_scale_factor`.
        """
        if self.recolor_cr:
            self.recolor_corneal_reflection(cr_parameters)
            base_image = self.cr_filled_image
            filter_function = not_on_ellipse
            x, y, r, a, b = cr_parameters
            filter_params = (x, y, r, self.cr_recolor_scale_factor*a,
                             self.cr_recolor_scale_factor*b)
        else:
            base_image = self.blurred_image
            filter_function = None
            filter_params = None
        return base_image, filter_function, filter_params

    def find_pupil(self, cr_parameters):
        """Estimate position of the pupil.

        Parameters
        ----------
        cr_parameters : numpy.ndarray
            [x, y, r, a, b] ellipse parameters of corneal reflection,
            used to prepare image if `recolor_cr` is set.

        Returns
        -------
        numpy.ndarray
            [x, y, r, a, b] ellipse parameters.
        """
        base_image, filter_function, filter_params = self.setup_pupil_finder(
            cr_parameters)
        seed_image = max_image_at_value(base_image,
                                        self.last_pupil_color)
        self.pupil_max_image = seed_image
        seed_point = max_convolution_positions(seed_image, self.pupil_mask,
                                               self.pupil_bounding_box)
        x, y, r, a, b = cr_parameters
        filter_params = (x, y, r, self.cr_recolor_scale_factor*a,
                          self.cr_recolor_scale_factor*b)
        candidate_points = self.point_generator.get_candidate_points(
            base_image, seed_point, True, filter_function=filter_function,
            filter_args=(filter_params, 2))
        return self.ellipse_fitter.fit(candidate_points)

    def recolor_corneal_reflection(self, cr_parameters):
        """Reshade the corneal reflection with the last pupil color.

        Parameters
        ----------
        cr_parameters : numpy.ndarray
            [x, y, r, a, b] ellipse parameters for corneal reflection.
        """
        x, y, r, a, b = cr_parameters
        a = self.cr_recolor_scale_factor*a + 1
        b = self.cr_recolor_scale_factor*b + 1
        r, c = ellipse_points((x, y, r, a, b), self.im_shape)
        self.cr_filled_image = self.blurred_image.copy()
        self.cr_filled_image[r,c] = self.last_pupil_color

    def update_last_pupil_color(self, pupil_parameters):
        """Update last pupil color with mean of fit.

        Parameters
        ----------
        pupil_parameters : numpy.ndarray
            [x, y, r, a, b] ellipse parameters for pupil.
        """
        if self.recolor_cr:
            image = self.cr_filled_image
        else:
            image = self.blurred_image
        r, c = ellipse_points(pupil_parameters, self.im_shape)
        value = int(np.mean(image[r, c]))
        value = max(self.min_pupil_value, value)
        value = min(self.max_pupil_value, value)
        self.last_pupil_color = value

    def process_image(self, image):
        self.current_image = image
        self.blurred_image = medfilt2d(image,
                                       kernel_size=SMOOTHING_KERNEL_SIZE)
        try:
            cr_parameters = self.find_corneal_reflection()
        except ValueError:
            logging.debug("Insufficient candidate points found for fitting "
                          "corneal reflection at frame %s", self.frame_index)
            cr_parameters = (np.nan, np.nan, np.nan, np.nan, np.nan)
        try:
            pupil_parameters = self.find_pupil(cr_parameters)
            self.update_last_pupil_color(pupil_parameters)
        except ValueError:
            logging.debug("Insufficient candidate points found for fitting "
                          "pupil at frame %s", self.frame_index)
            pupil_parameters = (np.nan, np.nan, np.nan, np.nan, np.nan)
        return cr_parameters, pupil_parameters

    def process_stream(self, n_frames=None, update_mean_frame=True):
        self.pupil_parameters = []
        self.cr_parameters = []
        if update_mean_frame:
            mean_frame = np.zeros(self.im_shape, dtype=np.float64)
        if n_frames is None:
            n_frames = self.input_stream.n_frames
        for i, frame in enumerate(self.input_stream):
            if update_mean_frame:
                mean_frame += frame
            self.frame_index = i
            cr_parameters, pupil_parameters = self.process_image(frame)
            self.cr_parameters.append(cr_parameters)
            self.pupil_parameters.append(pupil_parameters)
            if self.annotator.output_stream is not None:
                self.annotator.annotate_frame(frame, pupil_parameters,
                                              cr_parameters)
            if self.generate_QC_output:
                self.annotator.compute_density(frame, pupil_parameters,
                                               cr_parameters)
            self.annotator.clear_rc()
            if i == n_frames-1:
                break

        if self.annotator is not None:
            self.annotator.close()

        if update_mean_frame:
            self._mean_frame = (mean_frame / (i+1)).astype(np.uint8)

        return np.array(self.pupil_parameters), np.array(self.cr_parameters)


def default_bounding_box(image_shape):
    """Calculate a default bounding box as 10% in from borders of image.

    Parameters
    ----------
    image_shape : tuple
        (height, width) of image.

    Returns
    -------
    numpy.ndarray
        [xmin, xmax, ymin, ymax] bounding box.
    """
    h, w = image_shape
    x_crop = int(0.1*w)
    y_crop = int(0.1*h)

    return np.array([x_crop, w-x_crop, y_crop, h-y_crop], dtype='int')
