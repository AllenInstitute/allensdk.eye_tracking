import numpy as np


class RansacFitter(object):
    def __init__(self):
        self.best_error = np.inf
        self.best_params = None

    def fit(self, fit_function, error_function, data, threshold,
            minimum_points_for_fit, number_of_close_points, iterations):
        """Find a best fit to a model using ransac.

        Parameters
        ----------
        fit_function : callable
            Method that fits a model to `data`.
        error_function : callable
            Function that calculates error given `parameters` and a
            subset of `data`. Returns array of errors, one for each
            sample.
        data : numpy.ndarray
            Matrix of data points of shape (#samples, #items/sample).
        threshold : float
            Error threshold below which data should be considered an
            inlier.
        minimum_points_for_fit : int
            Number of points required to fit data.
        number_of_close_points : int
            Number of candidate outliers reselected as inliers required
            to consider a good fit.
        iterations : int
            Number of iterations to run.

        Returns
        -------
        Best model parameters, or None if no good fit found.

        Raises
        ------
        ValueError:
            If there is less data than `minimum_points_for_fit`.
        """
        if data.shape[0] < minimum_points_for_fit:
            raise ValueError("Insufficient data for fit")
        self.best_error = np.inf
        self.best_params = None
        for i in range(iterations):
            parameters, error = fit_iteration(fit_function, error_function,
                                              data, threshold,
                                              minimum_points_for_fit,
                                              number_of_close_points)
            if error < self.best_error:
                self.best_params = parameters
                self.best_error = error
        return self.best_params


def fit_iteration(fit_function, error_function, data, threshold,
                  minimum_points_for_fit, number_of_close_points):
    """Perform one iteration of ransac model fitting.

    Parameters
    ----------
    fit_function : callable
        Method that fits a model to `data`.
    error_function : callable
        Function that calculates error given `parameters` and a
        subset of `data`. Returns array of errors, one for each
        sample.
    data : numpy.ndarray
        Matrix of data points of shape (#samples, #items/sample).
    threshold : float
        Error threshold below which data should be considered an
        inlier.
    minimum_points_for_fit : int
        Number of points required to fit data.
    number_of_close_points : int
        Number of candidate outliers reselected as inliers required
        to consider a good fit.

    Returns
    -------
    tuple
        (model parameters, model error)
    """
    inlier_idx, outlier_idx = partition_candidate_indices(
        data, minimum_points_for_fit)
    parameters, error = fit_function(data[inlier_idx, :])
    if parameters is not None:
        also_inlier_idx = check_outliers(error_function, parameters, data,
                                         outlier_idx, threshold)
        if len(also_inlier_idx) > number_of_close_points:
            idx = np.concatenate((inlier_idx, also_inlier_idx))
            parameters, error = fit_function(data[idx, :])
            return parameters, error
    return None, np.inf


def check_outliers(error_function, parameters, data, outlier_indices,
                   threshold):
    """Check if any outliers should be inliers based on initial fit.

    Parameters
    ----------
    error_function : callable
        Function that calculates error given `parameters` and a
        subset of `data`. Returns array of errors, one for each
        sample.
    parameters : numpy.ndarray
        Model parameters after some fit.
    data : numpy.ndarray
        Matrix of data points of shape (#samples, #items/sample).
    outlier_indices : numpy.ndarray
        Index array for initial outlier guess.
    threshold : float
        Error threshold below which data should be considered an
        inlier.

    Returns
    -------
    numpy.ndarray
        Index array of new inliers.
    """
    also_in_index = error_function(parameters,
                                   data[outlier_indices, :]) < threshold

    return outlier_indices[also_in_index]


def partition_candidate_indices(data, minimum_points_for_fit):
    """Generation indices to partition data into inliers/outliers.

    Parameters
    ----------
    data : np.ndarray
        Matrix of data points of shape (#samples, #items/sample).
    minimum_points_for_fit : int
        Minimum number of points required to attempt fit.

    Returns
    -------
    tuple
        (inliers, outliers) tuple of index arrays for potential
    """
    shuffled = np.random.permutation(np.arange(data.shape[0]))

    inliers = shuffled[:minimum_points_for_fit]
    outliers = shuffled[minimum_points_for_fit:]

    return inliers, outliers
