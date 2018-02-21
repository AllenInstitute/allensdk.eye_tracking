from allensdk.eye_tracking import fit_ellipse as fe
import numpy as np
import pytest
from mock import patch


def rotate_vector(y, x, theta):

    xp = x*np.cos(theta) - y*np.sin(theta)
    yp = x*np.sin(theta) + y*np.cos(theta)

    return yp, xp


def ellipse_points(a, b, x0, y0, rotation):
    x = np.linspace(-a, a, 200)
    yp = np.sqrt(b**2 - (b**2 / a**2)*x**2)
    ym = -yp
    yp, x1 = rotate_vector(yp, x, rotation)
    ym, x2 = rotate_vector(ym, x, rotation)
    x = np.hstack((x1, x2)) + x0
    y = np.hstack((yp, ym)) + y0
    return np.vstack((y, x)).T


@pytest.mark.parametrize("a,b,x0,y0,rotation", [
    (3.0, 2.0, 20, 30, np.pi/6)
])
def test_ellipse_fit(a, b, x0, y0, rotation):
    data = ellipse_points(a, b, x0, y0, rotation)
    fitter = fe.EllipseFitter(40, 40, 0.01, 50)
    x, y, angle, ax1, ax2 = fitter.fit(data)
    assert(np.abs(x - x0) < 0.0001)
    assert(np.abs(y - y0) < 0.0001)
    assert(np.abs(angle - np.degrees(rotation)) < 0.01)
    assert((np.abs(ax1-a) < 0.0001 and np.abs(ax2-b) < 0.0001) or
           (np.abs(ax1-b) < 0.0001 and np.abs(ax2-a) < 0.0001))
    with patch.object(fitter._fitter, "fit", return_value=None):
        res = fitter.fit(data)
        assert(np.all(np.isnan(res)))
    results = fitter.fit(data, max_radius=min(a, b))
    assert(np.all(np.isnan(results)))
    results = fitter.fit(data, max_eccentricity=-1)
    assert(np.all(np.isnan(results)))


@pytest.mark.parametrize("point,ellipse_params,tolerance,result", [
    ((0, 30.0), (0, 0, 0, 30.0, 5.0), 0.01, False),
    ((0, 30.0), (0, 0, 0, 30.5, 5.0), 0.01, True),
    ((7.0, 7.0), (5, 5, 45, np.sqrt(8.0), 9), 0.01, False),
    ((7.0, 7.0), (5, 5, 45, 9, 9), 0.01, True)
])
def test_not_on_ellipse(point, ellipse_params, tolerance, result):
    assert(fe.not_on_ellipse(point, ellipse_params, tolerance) == result)
