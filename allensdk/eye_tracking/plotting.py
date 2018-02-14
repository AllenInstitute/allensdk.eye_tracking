import os
import warnings
import numpy as np
from skimage.draw import ellipse, ellipse_perimeter, polygon_perimeter
import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class Annotator(object):
    """Class for annotating frames with ellipses.

    Parameters
    ----------
    output_stream : object
        Object that implements a `write` method that accepts ndarray
        frames as well as `open` and `close` methods.
    """
    COLORS = {"cr": (0, 0, 255),
              "pupil": (255, 0, 0)}

    def __init__(self, output_stream=None):
        self.output_stream = output_stream
        self.densities = {"pupil": None,
                          "cr": None}
        self.clear_rc()

    def initiate_cumulative_data(self, shape):
        """Initialize density arrays to zeros of the correct shape.

        Parameters
        ----------
        shape : tuple
            (height, width) to make the density arrays.
        """
        self.densities["cr"] = np.zeros(shape, dtype=float)
        self.densities["pupil"] = np.zeros(shape, dtype=float)

    def clear_rc(self):
        """Clear the cached row and column ellipse border points."""
        self._r = {"pupil": None,
                   "cr": None}
        self._c = {"pupil": None,
                   "cr": None}

    def update_rc(self, name, ellipse_parameters, shape):
        """Cache new row and column ellipse border points.

        Parameters
        ----------
        name : string
            "pupil" or "cr" to reference the correct object in the
            lookup table.
        ellipse_parameters : tuple
            Conic parameters of the ellipse.
        shape : tuple
            (height, width) shape of image used to generate ellipse
            border points at the right rows and columns.

        Returns
        -------
        cache_updated : bool
            Whether or not new values were cached.
        """
        if np.any(np.isnan(ellipse_parameters)):
            return False
        if self._r[name] is None:
            self._r[name], self._c[name] = ellipse_perimeter_points(
                ellipse_parameters, shape)
        return True

    def _annotate(self, name, rgb_frame, ellipse_parameters):
        if self.update_rc(name, ellipse_parameters, rgb_frame.shape[:2]):
            color_by_points(rgb_frame, self._r[name], self._c[name],
                            self.COLORS[name])

    def annotate_frame(self, frame, pupil_parameters, cr_parameters,
                       seed=None, pupil_candidates=None):
        """Annotate an image with ellipses for cr and pupil.

        If the annotator was initialized with an output stream, the
        frame will be written to the stream.

        Parameters
        ----------
        frame : numpy.ndarray
            Grayscale image to annotate.
        pupil_parameters : tuple
            (x, y, r, a, b) ellipse parameters for pupil.
        cr_parameters : tuple
            (x, y, r, a, b) ellipse parameters for corneal reflection.
        seed : tuple
            (y, x) seed point of pupil.
        pupil_candidates : list
            List of (y, x) candidate points used for the ellipse
            fit of the pupil.

        Returns
        -------
        rgb_frame : numpy.ndarray
            Color annotated frame.
        """
        rgb_frame = get_rgb_frame(frame)
        if not np.any(np.isnan(pupil_parameters)):
            self._annotate("pupil", rgb_frame, pupil_parameters)
        if not np.any(np.isnan(cr_parameters)):
            self._annotate("cr", rgb_frame, cr_parameters)

        if seed is not None:
            color_by_points(rgb_frame, seed[0], seed[1], (0, 255, 0))

        if pupil_candidates:
            arr = np.array(pupil_candidates)
            color_by_points(rgb_frame, arr[:, 0], arr[:, 1], (0, 255, 0))

        if self.output_stream is not None:
            self.output_stream.write(rgb_frame)
        return rgb_frame

    def _density(self, name, frame, ellipse_parameters):
        if self.update_rc(name, ellipse_parameters, frame.shape):
            self.densities[name][self._r[name], self._c[name]] += 1

    def compute_density(self, frame, pupil_parameters, cr_parameters):
        """Update the density maps with from the current frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame.
        pupil_parameters : tuple
            (x, y, r, a, b) ellipse parameters for pupil.
        cr_parameters : tuple
            (x, y, r, a, b) ellipse parameters for corneal reflection.
        """
        # TODO: rename this to update_density
        if self.densities["pupil"] is None:
            self.initiate_cumulative_data(frame.shape)
        self._density("pupil", frame, pupil_parameters)
        self._density("cr", frame, cr_parameters)

    def annotate_with_cumulative_pupil(self, frame, filename=None):
        """Annotate frame with all pupil ellipses from the density map.

        Parameters
        ----------
        frame : numpy.ndarray
            Grayscale frame to annotate.
        filename : string
            Filename to save annotated image to, if provided.

        Returns
        -------
        rgb_frame : numpy.ndarray
            Annotated color frame.
        """
        return annotate_with_cumulative(frame, self.densities["pupil"],
                                        (0, 0, 255), filename)

    def annotate_with_cumulative_cr(self, frame, filename=None):
        """Annotate frame with all cr ellipses from the density map.

        Parameters
        ----------
        frame : numpy.ndarray
            Grayscale frame to annotate.
        filename : string
            Filename to save annotated image to, if provided.

        Returns
        -------
        rgb_frame : numpy.ndarray
            Annotated color frame.
        """
        return annotate_with_cumulative(frame, self.densities["cr"],
                                        (255, 0, 0), filename)

    def close(self):
        """Close the output stream if it exists."""
        if self.output_stream is not None:
            self.output_stream.close()


def get_rgb_frame(frame):
    """Convert a grayscale frame to an RGB frame.

    If the frame passed in already has 3 channels, it is simply returned.

    Parameters
    ----------
    frame : numpy.ndarray
        Image frame.

    Returns
    -------
    rgb_frame : numpy.ndarray
        [height,width,3] RGB frame.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        rgb_frame = frame
    elif frame.ndim == 2:
        rgb_frame = np.dstack([frame, frame, frame])
    else:
        raise ValueError("Frame of shape {} is not valid".format(frame.shape))
    return rgb_frame


def annotate_with_cumulative(frame, density, rgb_vals=(255, 0, 0),
                             filename=None):
    """Annotate frame with all values from `density`.

    Parameters
    ----------
    frame : numpy.ndarray
        Grayscale frame to annotate.
    density : numpy.ndarray
        Array of the same shape as frame with non-zero values
        where the image should be annotated.
    rgb_vals : tuple
        (r, g, b) 0-255 color values for annotation.
    filename : string
        Filename to save annotated image to, if provided.

    Returns
    -------
    rgb_frame : numpy.ndarray
        Annotated color frame.
    """
    rgb_frame = get_rgb_frame(frame)
    if density is not None:
        mask = density > 0
        color_by_mask(rgb_frame, mask, rgb_vals)
    if filename is not None:
        plt.imsave(filename, rgb_frame)
    return rgb_frame


def annotate_with_box(image, bounding_box, rgb_vals=(255, 0, 0),
                      filename=None):
    """Annotate image with bounding box.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image to annotate.
    bounding_box : numpy.ndarray
        [xmin, xmax, ymin, ymax] bounding box.
    rgb_vals : tuple
        (r, g, b) 0-255 color values for annotation.
    filename : string
        Filename to save annotated image to, if provided.

    Returns
    -------
    rgb_image : numpy.ndarray
        Annotated color image.
    """
    rgb_image = get_rgb_frame(image)
    xmin, xmax, ymin, ymax = bounding_box
    r = np.array((ymin, ymin, ymax, ymax), dtype=int)
    c = np.array((xmin, xmax, xmax, xmin), dtype=int)
    rr, cc = polygon_perimeter(r, c, rgb_image.shape[:2])
    color_by_points(rgb_image, rr, cc, rgb_vals)
    if filename is not None:
        plt.imsave(filename, rgb_image)
    return rgb_image


def color_by_points(rgb_image, row_points, column_points,
                    rgb_vals=(255, 0, 0)):
    """Color image at points indexed by row and column vectors.

    The image is recolored in-place.

    Parameters
    ----------
    rgb_image : numpy.ndarray
        Color image to draw into.
    row_points : numpy.ndarray
        Vector of row indices to color in.
    column_points : numpy.ndarray
        Vector of column indices to color in.
    rgb_vals : tuple
        (r, g, b) 0-255 color values for annotation.
    """
    for i, value in enumerate(rgb_vals):
        rgb_image[row_points, column_points, i] = value


def color_by_mask(rgb_image, mask, rgb_vals=(255, 0, 0)):
    """Color image at points indexed by mask.

    The image is recolored in-place.

    Parameters
    ----------
    rgb_image : numpy.ndarray
        Color image to draw into.
    mask : numpy.ndarray
        Boolean mask of points to color in.
    rgb_vals : tuple
        (r, g, b) 0-255 color values for annotation.
    """
    for i, value in enumerate(rgb_vals):
        rgb_image[mask, i] = value


def ellipse_points(ellipse_params, image_shape):
    """Generate row, column indices for filled ellipse.

    Parameters
    ----------
    ellipse_params : tuple
        (x, y, r, a b) ellipse parameters.
    image_shape : tuple
        (height, width) shape of image.

    Returns
    -------
    row_points : numpy.ndarray
        Row indices for filled ellipse.
    column_points : numpy.ndarray
        Column indices for filled ellipse.
    """
    x, y, r, a, b = ellipse_params
    r = np.radians(-r)
    return ellipse(y, x, b, a, image_shape, r)


def ellipse_perimeter_points(ellipse_params, image_shape):
    """Generate row, column indices for ellipse perimeter.

    Parameters
    ----------
    ellipse_params : tuple
        (x, y, r, a b) ellipse parameters.
    image_shape : tuple
        (height, width) shape of image.

    Returns
    -------
    row_points : numpy.ndarray
        Row indices for ellipse perimeter.
    column_points : numpy.ndarray
        Column indices for ellipse perimeter.
    """
    x, y, r, a, b = ellipse_params
    r = np.radians(r)
    return ellipse_perimeter(int(y), int(x), int(b), int(a), r, image_shape)


def get_filename(output_folder, prefix, image_type):
    """Helper function to build image filename.

    Parameters
    ----------
    output_folder : string
        Folder for images.
    prefix : string
        Image filename without extension.
    image_type : string
        File extension for image (e.g. '.png').

    Returns
    -------
    filename : string
        Fill filename of image, or None if no output folder.
    """
    if output_folder:
        filename = prefix + image_type
        return os.path.join(output_folder, filename)
    return None


def plot_cumulative(pupil_density, cr_density, output_dir=None, show=False,
                    image_type=".png"):
    """Plot cumulative density of ellipse fits for cr and pupil.

    Parameters
    ----------
    pupil_density : numpy.ndarray
        Accumulated density of pupil perimeters.
    pupil_density : numpy.ndarray
        Accumulated density of cr perimeters.
    output_dir : string
        Output directory to store images. Images aren't saved if None
        is provided.
    show : bool
        Whether or not to call pyplot.show() after generating both
        plots.
    image_type : string
        Image extension for saving plots.
    """
    dens = np.log(1+pupil_density)
    plot_density(np.max(dens) - dens,
                 filename=get_filename(output_dir, "pupil_density",
                                       image_type),
                 title="pupil density",
                 show=False)
    dens = np.log(1+cr_density)
    plot_density(np.max(dens) - dens,
                 filename=get_filename(output_dir, "cr_density",
                                       image_type),
                 title="cr density",
                 show=show)


def plot_summary(pupil_params, cr_params, output_dir=None, show=False,
                 image_type=".png"):
    """Plot timeseries of various pupil and cr parameters.

    Generates plots of pupil and cr parameters against frame number.
    The plots include (x, y) position, angle, and (semi-minor,
    semi-major) axes seperately for pupil and cr, for a total of 6
    plots.

    Parameters
    ----------
    pupil_params : numpy.ndarray
        Array of pupil parameters at every frame.
    cr_params : numpy.ndarray
        Array of cr parameters at every frame.
    output_dir : string
        Output directory for storing saved images of plots.
    show : bool
        Whether or not to call pyplot.show() after generating the plots.
    image_type : string
        File extension to use if saving images to `output_dir`.
    """
    plot_timeseries(pupil_params.T[0], "pupil x", x2=pupil_params.T[1],
                    label2="pupil y", title="pupil position",
                    filename=get_filename(output_dir, "pupil_position",
                                          image_type),
                    show=False)
    plot_timeseries(cr_params.T[0], "cr x", x2=cr_params.T[1],
                    label2="pupil y", title="cr position",
                    filename=get_filename(output_dir, "cr_position",
                                          image_type),
                    show=False)
    plot_timeseries(pupil_params.T[3], "pupil axis1", x2=pupil_params.T[4],
                    label2="pupil axis2", title="pupil major/minor axes",
                    filename=get_filename(output_dir, "pupil_axes",
                                          image_type),
                    show=False)
    plot_timeseries(cr_params.T[3], "cr axis1", x2=cr_params.T[4],
                    label2="cr axis1", title="cr major/minor axes",
                    filename=get_filename(output_dir, "cr_axes",
                                          image_type),
                    show=False)
    plot_timeseries(pupil_params.T[2], "pupil angle", title="pupil angle",
                    filename=get_filename(output_dir, "pupil_angle",
                                          image_type),
                    show=False)
    plot_timeseries(cr_params.T[2], "cr angle", title="cr angle",
                    filename=get_filename(output_dir, "cr_angle",
                                          image_type),
                    show=show)


def plot_timeseries(x1, label1, x2=None, label2=None, title=None,
                    filename=None, show=False):
    """Helper function to plot up to 2 timeseries against index.

    Parameters
    ----------
    x1 : numpy.ndarray
        Array of values to plot.
    label1 : string
        Label for `x1` timeseries.
    x2 : numpy.ndarray
        Optional second array of values to plot.
    label2 : string
        Label for `x2` timeseries.
    title : string
        Title for the plot.
    filename : string
        Filename to save the plot to.
    show : bool
        Whether or not to call pyplot.show() after generating the plot.
    """
    fig, ax = plt.subplots(1)
    ax.plot(x1, label=label1)
    if x2 is not None:
        ax.plot(x2, label=label2)
    ax.set_xlabel('frame index')
    if title:
        ax.set_title(title)
    ax.legend()
    if filename is not None:
        fig.savefig(filename)
    if show:
        plt.show()


def plot_density(density, title=None, filename=None, show=False):
    """Plot cumulative density.

    Parameters
    ----------
    density : numpy.ndarray
        Accumulated 2-D density map to plot.
    title : string
        Title for the plot.
    filename : string
        Filename to save the plot to.
    show : bool
        Whether or not to call pyplot.show() after generating the plot.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(density, cmap="gray", interpolation="nearest")
    if title:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename)
    if show:
        plt.show()
