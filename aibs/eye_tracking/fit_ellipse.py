"""Module for ellipse fitting.

The algorithm for the actual fitting is  detailed at
http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html.
"""
import numpy as np
from .ransac import RansacFitter
import logging


CONSTRAINT_MATRIX = np.zeros([6, 6])
CONSTRAINT_MATRIX[0, 2] = 2.0
CONSTRAINT_MATRIX[2, 0] = 2.0
CONSTRAINT_MATRIX[1, 1] = -1.0


class EllipseFitter(object):
    """Wrapper class for performing ransac fitting of an ellipse.

    Parameters
    ----------
    minimum_points_for_fit : int
        Number of points required to fit data.
    number_of_close_points : int
        Number of candidate outliers reselected as inliers required
        to consider a good fit.
    threshold : float
        Error threshold below which data should be considered an
        inlier.
    iterations : int
            Number of iterations to run.
    """
    DEFAULT_MINIMUM_POINTS_FOR_FIT = 10
    DEFAULT_NUMBER_OF_CLOSE_POINTS = 4
    DEFAULT_THRESHOLD = 0.0001
    DEFAULT_ITERATIONS = 10

    def __init__(self, minimum_points_for_fit=DEFAULT_MINIMUM_POINTS_FOR_FIT,
                 number_of_close_points=DEFAULT_NUMBER_OF_CLOSE_POINTS,
                 threshold=DEFAULT_THRESHOLD, iterations=DEFAULT_ITERATIONS):
        self.minimum_points_for_fit = minimum_points_for_fit
        self.number_of_close_points = number_of_close_points
        self.threshold = threshold
        self.iterations = iterations
        self._fitter = RansacFitter()

    def fit(self, candidate_points):
        """Perform a fit on (y,x) points.

        Parameters
        ----------
        candidate_points : list
            List of (y,x) points that may fit on the ellipse.

        Returns
        -------
        ellipse_parameters : tuple
            (x, y, angle, semi_axis1, semi_axis2) ellipse parameters.
        """
        data = np.array(candidate_points)
        params = self._fitter.fit(fit_ellipse, fit_errors, data,
                                  self.threshold, self.minimum_points_for_fit,
                                  self.number_of_close_points, self.iterations)
        if params is not None:
            x, y = ellipse_center(params)
            angle = ellipse_angle_of_rotation(params)*180/np.pi
            ax1, ax2 = ellipse_axis_length(params)
            return (x, y, angle, ax1, ax2)
        else:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)


def fit_ellipse(data):
    """Fit an ellipse to data.

    Parameters
    ----------
    data : numpy.ndarray
        [n,2] array of (y,x) data points.

    Returns
    -------
    ellipse_parameters : tuple
        (x, y, angle, semi_axis1, semi_axis2) ellipse parameters.
    error : float
        Mean error of the fit.
    """
    try:
        y, x = data.T

        D = np.vstack([x*x, x*y, y*y, x, y, np.ones(len(y))])
        S = np.dot(D, D.T)

        M = np.dot(np.linalg.inv(S), CONSTRAINT_MATRIX)
        U, s, V = np.linalg.svd(M)

        params = U.T[0]
        error = np.dot(params, np.dot(S, params))/len(y)
    except Exception as e:
        logging.debug(e)  # figure out which exception this is catching
        params = None
        error = np.inf

    return params, error


def fit_errors(parameters, data):
    """Calculate the errors on each data point.

    Parameters
    ----------
    parameters : numpy.ndarray
        Paramaters of the fit ellipse model.
    data : numpy.ndarray
        [n,2] array of (y,x) points.

    Returns
    -------
    numpy.ndarray
        Squared error of the fit at each point in data.
    """
    y, x = data.T
    D = np.vstack([x*x, x*y, y*y, x, y, np.ones(len(y))])
    errors = (np.dot(parameters, D))**2

    return errors


def quadratic_parameters(conic_parameters):
    """Get quadratic ellipse coefficients from conic parameters.

    Calculation from http://mathworld.wolfram.com/Ellipse.html

    Parameters
    ----------
    conic_parameters : tuple
        (x, y, angle, semi_axis1, semi_axis2) ellipse parameters.

    Returns
    -------
    quadratic_parameters : tuple
        Polynomial parameters for the ellipse.
    """
    a = conic_parameters[0]
    b = conic_parameters[1]/2
    c = conic_parameters[2]
    d = conic_parameters[3]/2
    f = conic_parameters[4]/2
    g = conic_parameters[5]
    return (a, b, c, d, f, g)


def ellipse_center(parameters):
    """Calculate the center of the ellipse given the model parameters.

    Calculation from http://mathworld.wolfram.com/Ellipse.html

    Parameters
    ----------
    parameters : numpy.ndarray
        Parameters of the ellipse fit.

    Returns
    -------
    center : numpy.ndarray
        [x,y] center of the ellipse.
    """
    a, b, c, d, f, g = quadratic_parameters(parameters)
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(parameters):
    """Calculate the rotation of the ellipse given the model parameters.

    Calculation from http://mathworld.wolfram.com/Ellipse.html

    Parameters
    ----------
    parameters : numpy.ndarray
        Parameters of the ellipse fit.

    Returns
    -------
    rotation : float
        Rotation of the ellipse.
    """
    a, b, c, d, f, g = quadratic_parameters(parameters)
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length(parameters):
    """Calculate the semi-axes lengths of the ellipse.

    Calculation from http://mathworld.wolfram.com/Ellipse.html

    Parameters
    ----------
    parameters : numpy.ndarray
        Parameters of the ellipse fit.

    Returns
    -------
    semi_axes : numpy.ndarray
        Semi-axes of the ellipse.
    """
    a, b, c, d, f, g = quadratic_parameters(parameters)
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))

    down1 = min(.0000000001, down1)
    down2 = min(.0000000001, down2)

    res1 = np.sqrt(up/down1)
    res2 = np.sqrt(up/down2)
    return np.array([res1, res2])


def not_on_ellipse(point, ellipse_params, tolerance):
    """Function that tests if a point is not on an ellipse.

    Parameters
    ----------
    point : tuple
        (y, x) point.
    ellipse_params : numpy.ndarray
        Ellipse parameters to check against.
    tolerance : float
        Tolerance for determining point is on ellipse.

    Returns
    ------
    not_on : bool
        True if `point` is not within `tolerance` of the ellipse.
    """
    py, px = point
    x, y, r, a, b = ellipse_params
    r = np.radians(r)
    # get point in frame of unrotated ellipse at 0, 0
    tx = (px - x)*np.cos(-r) - (py-y)*np.sin(-r)
    ty = (px - x)*np.sin(-r) + (py-y)*np.cos(-r)
    err = np.abs((tx**2 / a**2) + (ty**2 / b**2) - 1)
    if err < tolerance:
        return False
    return True
