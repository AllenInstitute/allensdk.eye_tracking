import numpy as np
import cv2

_CIRCLE_TEMPLATES = {}


def get_circle_template(radius, fill=1, surround=0):
    """Get circular template for estimating center point.

    Returns a cached template if it has already been computed.

    Parameters
    ----------
    radius : int
        Radius in pixels of the circle to draw.

    Returns
    -------
    template : numpy.ndarray
        Circle template.
    """
    global _CIRCLE_TEMPLATES
    mask = _CIRCLE_TEMPLATES.get((radius, int(fill), int(surround)), None)
    if mask is None:
        Y, X = np.meshgrid(np.arange(-radius-3, radius+4),
                           np.arange(-radius-3, radius+4))
        mask = np.ones([2*radius + 7, 2*radius + 7], dtype=np.float)*surround
        circle = X**2 + Y**2 < radius**2
        mask[circle] = fill
        _CIRCLE_TEMPLATES[(int(radius), int(fill), int(surround))] = mask
    return mask


def max_correlation_positions(image, template, bounding_box=None,
                              reject_coords=None):
    """Correlate image with template and return the max location.

    Correlation is done with mode set to `mode` and method 'fft'. It
    is only performed over the image region within `bounding_box` if it
    is provided. The resulting coordinates are provided in the context
    of the original image.

    Parameters
    ----------
    image : numpy.ndarray
        Image over which to convolve the kernel.
    template : numpy.ndarray
        Kernel to convolve with the image.
    bounding_box : numpy.ndarray
        [xmin, xmax, ymin, ymax] bounding box on the image.
    reject_coords : tuple
        (r, c) coordinates to disallow as best fit.

    Returns
    -------
    max_position : tuple
        (y, x) mean location maximum of the convolution of the kernel
        with the image.
    """
    if bounding_box is None:
        cropped_image = image
        xmin = 0
        ymin = 0
    else:
        xmin, xmax, ymin, ymax = bounding_box
        cropped_image = image[ymin:ymax, xmin:xmax]

    corr = cv2.matchTemplate(cropped_image.astype(np.float32),
                             template.astype(np.float32),
                             cv2.TM_CCORR_NORMED)

    if reject_coords:
        r = reject_coords[0] - ymin - template.shape[0]
        c = reject_coords[1] - xmin - template.shape[1]
        idx = (r >= 0) & (c >= 0)
        corr[r[idx], c[idx]] = -np.inf

    _, _, _, max_loc = cv2.minMaxLoc(corr)

    y = int(max_loc[1] + template.shape[0]/2.0 + ymin)
    x = int(max_loc[0] + template.shape[1]/2.0 + xmin)

    return y, x
