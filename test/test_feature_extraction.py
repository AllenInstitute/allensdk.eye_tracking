from allensdk.eye_tracking import feature_extraction
import numpy as np
from skimage.draw import circle
import pytest


@pytest.fixture
def image():
    image = np.zeros((100, 100))
    image[circle(30, 30, 10, (100, 100))] = 1
    image[circle(60, 60, 10, (100, 100))] = 1

    return image


@pytest.mark.parametrize("radius", [
    5,
    10
])
def test_get_circle_mask(radius):
    mask = feature_extraction.get_circle_mask(radius)
    assert(mask.shape == (2*radius+1, 2*radius+1))


@pytest.mark.parametrize("value", [
    0,
    255,
    100
])
def test_max_image_at_value(value):
    image, _ = np.meshgrid(np.arange(256), np.arange(256))
    processed = feature_extraction.max_image_at_value(image, value)
    value_index = np.where(processed == processed.max())
    assert(np.all(image[value_index] == value))


@pytest.mark.parametrize("image,bounding_box", [
    (image(), None),
    (image(), (10, 45, 10, 45)),
    (image(), (45, 75, 45, 75))
])
def test_max_convolution_positions(image, bounding_box):
    kernel = feature_extraction.get_circle_mask(8)
    y, x = feature_extraction.max_convolution_positions(image, kernel,
                                                        bounding_box)
    if bounding_box is not None:
        assert(x > bounding_box[0] and x < bounding_box[1])
        assert(y > bounding_box[2] and y < bounding_box[3])
