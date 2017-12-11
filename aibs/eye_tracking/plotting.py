import logging
import os
import numpy as np
from skimage.draw import ellipse, ellipse_perimeter
from matplotlib import pyplot as plt


class Annotator(object):
    COLORS = {"cr": (0,0,255),
              "pupil": (255,0,0)}
    def __init__(self, output_stream=None):
        self.output_stream = output_stream
        self.densities = {"pupil": None,
                          "cr": None}
        self.clear_rc()

    def initiate_cumulative_data(self, shape):
        self.densities["cr"] = np.zeros(shape, dtype=float)
        self.densities["pupil"] = np.zeros(shape, dtype=float)

    def clear_rc(self):
        # reset cache for last set of calculated border points
        self._r = {"pupil": None,
                   "cr": None}
        self._c = {"pupil": None,
                   "cr": None}

    def update_rc(self, name, ellipse_parameters, shape):
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

    def annotate_frame(self, frame, pupil_parameters, cr_parameters):
        rgb_frame = np.dstack([frame, frame, frame])
        if not np.any(np.isnan(pupil_parameters)):
            self._annotate("pupil", rgb_frame, pupil_parameters)
        if not np.any(np.isnan(cr_parameters)):
            self._annotate("cr", rgb_frame, cr_parameters)

        if self.output_stream is not None:
            self.output_stream.write(rgb_frame)
        return rgb_frame

    def _density(self, name, frame, ellipse_parameters):
        if self.update_rc(name, ellipse_parameters, frame.shape):
            self.densities[name][self._r[name], self._c[name]] += 1

    def compute_density(self, frame, pupil_parameters, cr_parameters):
        if self.densities["pupil"] is None:
            self.initiate_cumulative_data(frame.shape)
        self._density("pupil", frame, pupil_parameters)
        self._density("cr", frame, cr_parameters)

    def annotate_with_cumulative_pupil(self, frame, filename=None):
        return annotate_with_cumulative(frame, self.densities["pupil"],
                                        (0,0,255), filename)

    def annotate_with_cumulative_cr(self, frame, filename=None):
        return annotate_with_cumulative(frame, self.densities["cr"], (255,0,0),
                                        filename)

    def close(self):
        self.output_stream.close()


def annotate_with_cumulative(frame, density, rgb_vals=(255,0,0),
                             filename=None):
    rgb_frame = np.dstack([frame, frame, frame])
    if density is not None:
        mask = density > 0
        color_by_mask(rgb_frame, mask, rgb_vals)
    if filename is not None:
        plt.imsave(filename, rgb_frame)
    return rgb_frame


def color_by_points(rgb_image, row_points, column_points, rgb_vals=(255,0,0)):
    for i, value in enumerate(rgb_vals):
        rgb_image[row_points, column_points, i] = value


def color_by_mask(rgb_image, mask, rgb_vals=(255,0,0)):
    for i, value in enumerate(rgb_vals):
        rgb_image[mask, i] = value


def ellipse_points(ellipse_params, image_shape):
    x, y, r, a, b = ellipse_params
    r = np.radians(-r)
    return ellipse(y, x, b, a, image_shape, r)


def ellipse_perimeter_points(ellipse_params, image_shape):
    x, y, r, a, b = ellipse_params
    r = np.radians(r)
    return ellipse_perimeter(int(y), int(x), int(b), int(a), r, image_shape)


def get_filename(output_folder, prefix, image_type):
    if output_folder:
        filename = prefix + image_type
        return os.path.join(output_folder, filename)
    return None


def plot_cumulative(pupil_density, cr_density, image=None, output_dir=None,
                    show=False, image_type=".png"):
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
    fig, ax = plt.subplots(1)
    ax.imshow(density, cmap="gray", interpolation="nearest")
    if title:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename)
    if show:
        plt.show()
