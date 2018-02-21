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
def test_get_circle_template(radius):
    mask = feature_extraction.get_circle_template(radius)
    assert(mask.shape == (2*radius+7, 2*radius+7))


@pytest.mark.parametrize("image,bounding_box", [
    (image(), None),
    (image(), (10, 45, 10, 45)),
    (image(), (45, 75, 45, 75))
])
def test_max_correlation_positions(image, bounding_box):
    kernel = feature_extraction.get_circle_template(8)
    y, x = feature_extraction.max_correlation_positions(image, kernel,
                                                        bounding_box)
    if bounding_box is not None:
        assert(x > bounding_box[0] and x < bounding_box[1])
        assert(y > bounding_box[2] and y < bounding_box[3])
