import logging
import cv2
import numpy as np
from .fit_ellipse import EllipseFitter, ellipse_pass_filter
from .utils import generate_ray_indices, get_ray_values
from .feature_extraction import (get_circle_template,
                                 max_correlation_positions)
from .plotting import Annotator, ellipse_points


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
    cr_threshold_factor : float
        Multiplicative factor for thresholding corneal reflection.
    pupil_threshold_factor : float
        Multiplicative factor for thresholding pupil.
    cr_threshold_pixels : int
        Number of pixels (from beginning of ray) to use to determine
        threshold of corneal reflection.
    pupil_threshold_pixels : int
        Number of pixels (from beginning of ray) to use to determine
        threshold of pupil.
    """
    DEFAULT_INDEX_LENGTH = 150
    DEFAULT_N_RAYS = 150
    DEFAULT_THRESHOLD_FACTOR = 1.2
    DEFAULT_CR_THRESHOLD_PIXELS = 10
    DEFAULT_PUPIL_THRESHOLD_PIXELS = 22

    def __init__(self, index_length=DEFAULT_INDEX_LENGTH,
                 n_rays=DEFAULT_N_RAYS,
                 cr_threshold_factor=DEFAULT_THRESHOLD_FACTOR,
                 pupil_threshold_factor=DEFAULT_THRESHOLD_FACTOR,
                 cr_threshold_pixels=DEFAULT_CR_THRESHOLD_PIXELS,
                 pupil_threshold_pixels=DEFAULT_PUPIL_THRESHOLD_PIXELS):
        self.update_params(index_length=index_length, n_rays=n_rays,
                           cr_threshold_factor=cr_threshold_factor,
                           pupil_threshold_factor=pupil_threshold_factor,
                           cr_threshold_pixels=cr_threshold_pixels,
                           pupil_threshold_pixels=pupil_threshold_pixels)
        self.above_threshold = {"cr": False,
                                "pupil": True}
        self._intensity_estimate = 0

    def update_params(self, index_length=DEFAULT_INDEX_LENGTH,
                      n_rays=DEFAULT_N_RAYS,
                      cr_threshold_factor=DEFAULT_THRESHOLD_FACTOR,
                      pupil_threshold_factor=DEFAULT_THRESHOLD_FACTOR,
                      cr_threshold_pixels=DEFAULT_CR_THRESHOLD_PIXELS,
                      pupil_threshold_pixels=DEFAULT_PUPIL_THRESHOLD_PIXELS):
        """Update starburst point generation parameters.

        Parameters
        ----------
        index_length : int
            Initial default length for ray indices.
        n_rays : int
            The number of rays to check.
        cr_threshold_factor : float
            Multiplicative factor for thresholding corneal reflection.
        pupil_threshold_factor : float
            Multiplicative factor for thresholding pupil.
        cr_threshold_pixels : int
            Number of pixels (from beginning of ray) to use to determine
            threshold of corneal reflection.
        pupil_threshold_pixels : int
            Number of pixels (from beginning of ray) to use to determine
            threshold of pupil.
        """
        self.index_length = index_length
        self.xs, self.ys = generate_ray_indices(index_length, n_rays)
        self.threshold_pixels = {"cr": cr_threshold_pixels,
                                 "pupil": pupil_threshold_pixels}
        self.threshold_factor = {"cr": cr_threshold_factor,
                                 "pupil": pupil_threshold_factor}

    def get_candidate_points(self, image, seed_point, point_type,
                             filter_function=None, filter_args=(),
                             filter_kwargs=None):
        """Get candidate points for ellipse fitting.

        Parameters
        ----------
        image : numpy.ndarray
            Image to check for threshold crossings.
        seed_point : tuple
            (y, x) center point for ray burst.
        point_type : str
            Either 'cr' or 'pupil'. Determines if threshold crossing is
            high-to-low or low-to-high and which `threshold_factor` and
            `threshold_pixels` value to use.

        Returns
        -------
        candidate_points : list
            List of (y, x) candidate points.
        """
        xs = self.xs + seed_point[1]
        ys = self.ys + seed_point[0]
        ray_values = get_ray_values(xs, ys, image)
        filtered_out = 0
        threshold_not_crossed = 0
        candidate_points = []
        if filter_kwargs is None:
            filter_kwargs = {}
        for i, values in enumerate(ray_values):
            try:
                point = self.threshold_crossing(
                    xs[i], ys[i], values, point_type)
                if filter_function is not None:
                    filter_kwargs["pupil_intensity_estimate"] = \
                        self._intensity_estimate
                    if filter_function(point, *filter_args, **filter_kwargs):
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

    def threshold_crossing(self, xs, ys, values, point_type):
        """Check a ray for where it crosses a threshold.

        The threshold is calculated using `get_threshold`.

        Parameters
        ----------
        xs : numpy.ndarray
            X indices of ray.
        ys : numpy.ndarray
            Y indices of ray.
        values : numpy.ndarray
            Image values along ray.
        point_type : str
            Either 'cr' or 'pupil'. Determines if threshold crossing is
            high-to-low or low-to-high and which `threshold_factor` and
            `threshold_pixels` value to use.

        Returns
        -------
        y_index : int
            Y index of threshold crossing.
        x_index : int
            X index of threshold crossing.

        Raises
        ------
        ValueError
            If no threshold crossing found.
        """
        try:
            above_threshold = self.above_threshold[point_type]
            threshold_pixels = self.threshold_pixels[point_type]
            threshold_factor = self.threshold_factor[point_type]
        except KeyError:
            raise ValueError(("'{}' is not a supported point type, must be "
                              "'cr' or 'pupil'").format(point_type))
        threshold = self.get_threshold(values, threshold_pixels,
                                       threshold_factor)
        if above_threshold:
            comparison = values[threshold_pixels:] > threshold
        else:
            comparison = values[threshold_pixels:] < threshold
        sub_index = np.argmax(comparison)
        if comparison[sub_index]:
            index = threshold_pixels + sub_index
            return ys[index], xs[index]
        else:
            raise ValueError("No value in array crosses: {}".format(threshold))

    def get_threshold(self, ray_values, threshold_pixels, threshold_factor):
        """Calculate the threshold from the ray values.

        The threshold is determined from `threshold_factor` times the
        mean of the first `threshold_pixels` values.

        Parameters
        ----------
        ray_values : numpy.ndarray
            Values of the ray.
        threshold_factor : float
            Multiplicative factor for thresholding.
        threshold_pixels : int
            Number of pixels (from beginning of ray) to use to determine
            threshold.

        Returns
        -------
        threshold : float
            Threshold to set for candidate point.
        """
        sub_ray = ray_values[threshold_pixels]
        self._intensity_estimate = np.mean(sub_ray)
        threshold = threshold_factor*self._intensity_estimate

        return threshold


class EyeTracker(object):
    """Mouse Eye-Tracker.

    Parameters
    ----------
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
        adaptive_pupil: bool
        smoothing_kernel_size : int
        clip_pupil_values : bool
        average_iris_intensity : int
    """
    DEFAULT_MIN_PUPIL_VALUE = 0
    DEFAULT_MAX_PUPIL_VALUE = 40
    DEFAULT_CR_RECOLOR_SCALE_FACTOR = 1.7
    DEFAULT_RECOLOR_CR = True
    DEFAULT_ADAPTIVE_PUPIL = False
    DEFAULT_CR_MASK_RADIUS = 10
    DEFAULT_PUPIL_MASK_RADIUS = 35
    DEFAULT_GENERATE_QC_OUTPUT = False
    DEFAULT_SMOOTHING_KERNEL_SIZE = 7
    DEFAULT_CLIP_PUPIL_VALUES = True
    DEFAULT_AVERAGE_IRIS_INTENSITY = 40
    DEFAULT_MAX_ECCENTRICITY = 0.25

    def __init__(self, input_stream, output_stream=None,
                 starburst_params=None, ransac_params=None,
                 pupil_bounding_box=None, cr_bounding_box=None,
                 generate_QC_output=DEFAULT_GENERATE_QC_OUTPUT, **kwargs):
        self._mean_frame = None
        self._input_stream = None
        self.input_stream = input_stream
        self.point_generator = None
        self.ellipse_fitter = None
        self.min_pupil_value = self.DEFAULT_MIN_PUPIL_VALUE
        self.max_pupil_value = self.DEFAULT_MAX_PUPIL_VALUE
        self.cr_recolor_scale_factor = self.DEFAULT_CR_RECOLOR_SCALE_FACTOR
        self.recolor_cr = self.DEFAULT_RECOLOR_CR
        self.cr_mask_radius = self.DEFAULT_CR_MASK_RADIUS
        self.pupil_mask_radius = self.DEFAULT_PUPIL_MASK_RADIUS
        self.adaptive_pupil = self.DEFAULT_ADAPTIVE_PUPIL
        self.smoothing_kernel_size = self.DEFAULT_SMOOTHING_KERNEL_SIZE
        self.clip_pupil_values = self.DEFAULT_CLIP_PUPIL_VALUES
        self.average_iris_intensity = self.DEFAULT_AVERAGE_IRIS_INTENSITY
        self.max_eccentricity = self.DEFAULT_MAX_ECCENTRICITY
        self.update_fit_parameters(starburst_params=starburst_params,
                                   ransac_params=ransac_params,
                                   pupil_bounding_box=pupil_bounding_box,
                                   cr_bounding_box=cr_bounding_box,
                                   **kwargs)
        self.annotator = Annotator(output_stream)
        self.pupil_parameters = []
        self.cr_parameters = []
        self.pupil_colors = []
        self.generate_QC_output = generate_QC_output
        self.current_seed = None
        self.current_pupil_candidates = None
        self.current_image = None
        self.current_image_mean = 0
        self.blurred_image = None
        self.cr_filled_image = None
        self.pupil_max_image = None
        self.annotated_image = None
        self.frame_index = 0

    def update_fit_parameters(self, starburst_params=None, ransac_params=None,
                              pupil_bounding_box=None, cr_bounding_box=None,
                              **kwargs):
        """Update EyeTracker fitting parameters.

        Parameters
        ----------
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
            adaptive_pupil: bool
            smoothing_kernel_size : int
            clip_pupil_values : bool
            average_iris_intensity : int
        """
        if self.point_generator is None:
            if starburst_params is None:
                self.point_generator = PointGenerator()
            else:
                self.point_generator = PointGenerator(**starburst_params)
        elif starburst_params is not None:
            self.point_generator.update_params(**starburst_params)
        if self.ellipse_fitter is None:
            if ransac_params is None:
                self.ellipse_fitter = EllipseFitter()
            else:
                self.ellipse_fitter = EllipseFitter(**ransac_params)
        elif ransac_params is not None:
            self.ellipse_fitter.update_params(**ransac_params)
        if pupil_bounding_box is None or len(pupil_bounding_box) != 4:
            pupil_bounding_box = default_bounding_box(self.im_shape)
        if cr_bounding_box is None or len(cr_bounding_box) != 4:
            cr_bounding_box = default_bounding_box(self.im_shape)
        self.pupil_bounding_box = pupil_bounding_box
        self.cr_bounding_box = cr_bounding_box
        self._init_kwargs(**kwargs)
        self.current_seed = None
        self.current_pupil_candidates = None
        self.current_image = None
        self.current_image_mean = 0
        self.blurred_image = None
        self.cr_filled_image = None
        self.annotated_image = None

    def _init_kwargs(self, **kwargs):
        self.min_pupil_value = kwargs.get("min_pupil_value",
                                          self.min_pupil_value)
        self.max_pupil_value = kwargs.get("max_pupil_value",
                                          self.max_pupil_value)
        self.last_pupil_color = self.min_pupil_value
        self.cr_recolor_scale_factor = kwargs.get(
            "cr_recolor_scale_factor", self.cr_recolor_scale_factor)
        self.recolor_cr = kwargs.get("recolor_cr", self.recolor_cr)
        self.cr_mask_radius = kwargs.get("cr_mask_radius", self.cr_mask_radius)
        self.cr_mask = get_circle_template(self.cr_mask_radius, fill=1,
                                           surround=-1)
        self.pupil_mask_radius = kwargs.get("pupil_mask_radius",
                                            self.pupil_mask_radius)
        self.adaptive_pupil = kwargs.get(
            "adaptive_pupil", self.adaptive_pupil)
        self.smoothing_kernel_size = kwargs.get(
            "smoothing_kernel_size", self.smoothing_kernel_size)
        self.clip_pupil_values = kwargs.get(
            "clip_pupil_values", self.clip_pupil_values)
        if self.clip_pupil_values:
            self.pupil_limits = (self.min_pupil_value,
                                 self.max_pupil_value)
        else:
            self.pupil_limits = None
        self.average_iris_intensity = kwargs.get(
            "average_iris_intensity", self.average_iris_intensity)
        self.max_eccentricity = kwargs.get(
            "max_eccentricity", self.max_eccentricity)

    @property
    def im_shape(self):
        """Image shape."""
        if self.input_stream is None:
            return None
        return self.input_stream.frame_shape

    @property
    def input_stream(self):
        """Input frame source."""
        return self._input_stream

    @input_stream.setter
    def input_stream(self, stream):
        self._mean_frame = None
        if self._input_stream is not None:
            self._input_stream.close()
        if stream is not None and stream.frame_shape != self.im_shape:
            self.cr_bounding_box = default_bounding_box(stream.frame_shape)
            self.pupil_bounding_box = default_bounding_box(stream.frame_shape)
        self._input_stream = stream

    @property
    def mean_frame(self):
        """Average frame calculated from the input source."""
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
        ellipse_parameters : tuple
            (x, y, r, a, b) ellipse parameters.
        """
        seed_point = max_correlation_positions(
            self.blurred_image, self.cr_mask, self.cr_bounding_box)
        candidate_points = self.point_generator.get_candidate_points(
            self.blurred_image, seed_point, "cr")
        return self.ellipse_fitter.fit(
            candidate_points, max_radius=self.point_generator.index_length)

    def setup_pupil_finder(self, cr_parameters):
        """Initialize image and ransac filter for pupil fitting.

        If recoloring the corneal_reflection, color it in and provide a
        filter to exclude points that fall on the colored-in ellipse
        from fitting.

        Parameters
        ----------
        cr_parameters : tuple
            (x, y, r, a, b) ellipse parameters for corneal reflection.

        Returns
        -------
        image : numpy.ndarray
            Image for pupil fitting. Has corneal reflection filled in if
            `recolor_cr` is set.
        filter_function : callable
            Function to indicate if points fall on the recolored ellipse
            or None if not recoloring.
        filter_parameters : tuple
            Ellipse parameters for recolor ellipse shape, which are
            `cr_parameters` with the axes scaled by
            `cr_recolor_scale_factor`.
        """
        if self.recolor_cr:
            self.recolor_corneal_reflection(cr_parameters)
            base_image = self.cr_filled_image
            filter_function = ellipse_pass_filter
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
        cr_parameters : tuple
            (x, y, r, a, b) ellipse parameters of corneal reflection,
            used to prepare image if `recolor_cr` is set.

        Returns
        -------
        ellipse_parameters : tuple
            (x, y, r, a, b) ellipse parameters.
        """
        base_image, filter_function, filter_params = self.setup_pupil_finder(
            cr_parameters)
        pupil_mask = get_circle_template(
            self.pupil_mask_radius,
            int(self.last_pupil_color),
            int(self.average_iris_intensity))

        # template matching uses top-left corner for the best match, so shift
        # rejection coordinates accordingly
        if self.recolor_cr:
            reject = (self._recolored_r - int(pupil_mask.shape[0]/2.0),
                      self._recolored_c - int(pupil_mask.shape[1]/2.0))
        else:
            reject = None
        seed_point = max_correlation_positions(
            base_image, pupil_mask,
            self.pupil_bounding_box, reject_coords=reject)

        filter_kwargs = {}
        if self.clip_pupil_values:
            filter_kwargs = {"pupil_limits": self.pupil_limits}

        candidate_points = self.point_generator.get_candidate_points(
            base_image, seed_point, "pupil", filter_function=filter_function,
            filter_args=(filter_params, 2), filter_kwargs=filter_kwargs)
        self.current_seed = seed_point
        self.current_pupil_candidates = candidate_points

        return self.ellipse_fitter.fit(
            candidate_points, max_radius=self.point_generator.index_length,
            max_eccentricity=self.max_eccentricity)

    def recolor_corneal_reflection(self, cr_parameters):
        """Reshade the corneal reflection with the last pupil color.

        Parameters
        ----------
        cr_parameters : tuple
            (x, y, r, a, b) ellipse parameters for corneal reflection.
        """
        x, y, r, a, b = cr_parameters
        a = self.cr_recolor_scale_factor*a + 1
        b = self.cr_recolor_scale_factor*b + 1
        r, c = ellipse_points((x, y, r, a, b), self.blurred_image.shape)
        self.cr_filled_image = self.blurred_image.copy()
        self.cr_filled_image[r, c] = self.last_pupil_color
        self._recolored_r = r
        self._recolored_c = c

    def update_last_pupil_color(self, pupil_parameters):
        """Update last pupil color with mean of fit.

        Parameters
        ----------
        pupil_parameters : tuple
            (x, y, r, a, b) ellipse parameters for pupil.
        """
        if np.any(np.isnan(pupil_parameters)):
            return
        if self.recolor_cr:
            image = self.cr_filled_image
        else:
            image = self.blurred_image
        r, c = ellipse_points(pupil_parameters, image.shape)
        value = int(np.mean(image[r, c]))
        value = max(self.min_pupil_value, value)
        value = min(self.max_pupil_value, value)
        self.last_pupil_color = value

    def process_image(self, image):
        """Process an image to find pupil and corneal reflection.

        Parameters
        ----------
        image : numpy.ndarray
            Image to process.

        Returns
        -------
        cr_parameters : tuple
            (x, y, r, a, b) corneal reflection parameters.
        pupil_parameters : tuple
            (x, y, r, a, b) pupil parameters.
        cr_error : float
            Ellipse fit error for best fit.
        pupil_error : float
            Ellipse fit error for best fit.
        """
        self.current_image = image
        self.current_image_mean = self.current_image.mean()
        self.blurred_image = cv2.medianBlur(image, self.smoothing_kernel_size)
        try:
            cr_parameters, cr_error = self.find_corneal_reflection()
        except ValueError:
            logging.debug("Insufficient candidate points found for fitting "
                          "corneal reflection at frame %s", self.frame_index)
            cr_parameters = (np.nan, np.nan, np.nan, np.nan, np.nan)
            cr_error = np.nan

        try:
            pupil_parameters, pupil_error = self.find_pupil(cr_parameters)
            if self.adaptive_pupil:
                self.update_last_pupil_color(pupil_parameters)
        except ValueError:
            logging.debug("Insufficient candidate points found for fitting "
                          "pupil at frame %s", self.frame_index)
            pupil_parameters = (np.nan, np.nan, np.nan, np.nan, np.nan)
            pupil_error = np.nan

        return cr_parameters, pupil_parameters, cr_error, pupil_error

    def process_stream(self, start=0, stop=None, step=1,
                       update_mean_frame=True):
        """Get cr and pupil parameters from frames of `input_stream`.

        By default this will process every frame in the input stream.

        Parameters
        ----------
        start : int
            Index of first frame to process. Defaults to 0.
        stop : int
            Stop index for processing. Defaults to None, which runs
            runs until the end of the input stream.
        step : int
            Number of frames to advance at each iteration. Used to skip
            frames while processing. Set to 1 to process every frame, 2
            to process every other frame, etc. Defaults to 1.
        update_mean_frame : bool
            Whether or not to update the mean frame while processing
            the frames.

        Returns
        -------
        cr_parameters : numpy.ndarray
            [n_frames,5] array of corneal reflection parameters.
        pupil_parameters : numpy.ndarray
            [n_frames,5] array of pupil parameters.
        cr_errors : numpy.ndarray
            [n_frames,] array of fit errors for corneal reflection
            ellipses.
        pupil_errors : numpy.ndarray
            [n_frames,] array of fit errors for pupil ellipses.
        """
        self.pupil_parameters = []
        self.cr_parameters = []
        self.pupil_errors = []
        self.cr_errors = []
        self.pupil_colors = []
        i = 0

        if update_mean_frame:
            mean_frame = np.zeros(self.im_shape, dtype=np.float64)

        for i, frame in enumerate(self.input_stream[start:stop:step]):
            if update_mean_frame:
                mean_frame += frame
            self.frame_index = start + step*i
            cr_parameters, pupil_parameters, cr_error, pupil_error = \
                self.process_image(frame)
            self.cr_parameters.append(cr_parameters)
            self.pupil_parameters.append(pupil_parameters)
            self.cr_errors.append(cr_error)
            self.pupil_errors.append(pupil_error)
            self.pupil_colors.append(self.last_pupil_color)
            if self.annotator.output_stream is not None:
                self.annotated_image = self.annotator.annotate_frame(
                    frame, pupil_parameters, cr_parameters, self.current_seed,
                    self.current_pupil_candidates)
            if self.generate_QC_output:
                self.annotator.compute_density(frame, pupil_parameters,
                                               cr_parameters)
            self.annotator.clear_rc()

        self.annotator.close()

        if update_mean_frame:
            self._mean_frame = (mean_frame / (i+1)).astype(np.uint8)

        return (np.array(self.cr_parameters), np.array(self.pupil_parameters),
                np.array(self.cr_errors), np.array(self.pupil_errors))


def default_bounding_box(image_shape):
    """Calculate a default bounding box as 10% in from borders of image.

    Parameters
    ----------
    image_shape : tuple
        (height, width) of image.

    Returns
    -------
    bounding_box : numpy.ndarray
        [xmin, xmax, ymin, ymax] bounding box.
    """
    if image_shape is None:
        return np.array([1, -1, 1, -1], dtype='int')

    h, w = image_shape
    x_crop = int(0.1*w)
    y_crop = int(0.1*h)

    return np.array([x_crop, w-x_crop, y_crop, h-y_crop], dtype='int')
