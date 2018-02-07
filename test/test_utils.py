from allensdk.eye_tracking import utils
import numpy as np
import pytest


@pytest.fixture
def image():
    im, _ = np.meshgrid(np.arange(400), np.arange(400))
    return im


@pytest.mark.parametrize("index_length,n_rays", [
    (200, 30),
    (100, 10)
])
def test_rotate_rays(index_length, n_rays):
    x = np.arange(index_length).reshape(1, index_length)
    y = np.zeros((1, index_length))
    a = (np.arange(n_rays)*2.0*np.pi/n_rays).reshape(n_rays, 1)
    xr, yr = utils.rotate_rays(x, y, a)
    assert(xr.shape == yr.shape == (n_rays, index_length))
    with pytest.raises(ValueError):
        a = a.reshape(1, n_rays)
        utils.rotate_rays(x, y, a)


@pytest.mark.parametrize("index_length,n_rays", [
    (200, 30),
    (100, 10)
])
def test_generate_ray_indices(index_length, n_rays):
    xr, yr = utils.generate_ray_indices(index_length, n_rays)
    assert(xr.shape == yr.shape == (n_rays, index_length))


@pytest.mark.parametrize("index_length,n_rays,image", [
    (200, 20, image()),
    (600, 5, image())
])
def test_get_ray_values(index_length, n_rays, image):
    x, y = utils.generate_ray_indices(index_length, n_rays)
    values = utils.get_ray_values(x, y, image)
    assert(len(values) == n_rays)
