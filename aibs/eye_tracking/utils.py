import numpy as np
from scipy.signal import correlate2d
from scipy.signal import fftconvolve
from scipy.ndimage.filters import sobel
import logging


def good_coordinate_mask(xs, ys, shape):
    """Generate a coordinate mask inside `shape`.

    Parameters
    ----------
    xs : np.ndarray
        X indices.
    ys : np.ndarray
        Y indices.
    shape : tuple
        (height, width) shape of image.

    Returns
    -------
    np.ndarray
        Logical mask for `xs` and `xs` to ensure they are inside image.
    """
    return np.logical_and(np.logical_and(ys >= 0, ys < shape[0]),
                          np.logical_and(xs >= 0, xs < shape[1]))


def get_ray_values(xs, ys, image):
    """Get values of image along a set of rays.

    Parameters
    ----------
    xs : np.ndarray
        X indices of rays.
    ys : np.ndarray
        Y indices of rays.
    image : np.ndarray
        Image to get values from.

    Returns
    -------
    list
        List of arrays of image values along each ray.
    """
    ray_values = []
    for i in range(xs.shape[0]):
        mask = good_coordinate_mask(xs[i], ys[i], image.shape)
        xm = xs[i][mask]
        ym = ys[i][mask]
        ray_values.append(image[ym,xm])

    return ray_values


def generate_ray_indices(index_length, n_rays):
    """Generate index arrays for rays emanating in a circle from a point.

    Rays have start point at 0,0.

    Parameters
    ----------
    index_length : int
        Length of each index array.
    n_rays : int
        Number of rays to generate. Rays are evenly distributed about
        360 degrees.

    Returns
    -------
    tuple
        (xs, ys) tuple of [n_rays,index_length] index matrices.
    """
    angles = (np.arange(n_rays)*2.0*np.pi/n_rays).reshape(n_rays, 1)
    xs = np.arange(index_length).reshape(1, index_length)
    ys = np.zeros((1, index_length))

    return rotate_rays(xs, ys, angles)


def rotate_rays(xs, ys, angles):
    """Rotate index arrays about angles.

    Parameters
    ----------
    xs : np.ndarray
        Unrotated x-index array of shape [1,n].
    ys : np.ndarray
        Unrotated y-index array of shape [1,n].
    angles : np.adarray
        Angles over which to rotate of shape [m,1].

    Returns
    -------
    tuple
        (xs, ys) tuple of [m,n] index matrices.
    """
    cosines = np.cos(angles)
    sines = np.sin(angles)
    x_rot = np.dot(cosines, xs) + np.dot(sines, ys)
    y_rot = np.dot(cosines, ys) - np.dot(sines, xs)

    return x_rot.astype(np.int64), y_rot.astype(np.int64)
