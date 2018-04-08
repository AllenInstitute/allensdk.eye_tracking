[![Build Status](https://travis-ci.org/AllenInstitute/allensdk.eye_tracking.svg?branch=master)](https://travis-ci.org/AllenInstitute/allensdk.eye_tracking)
[![Build status](https://ci.appveyor.com/api/projects/status/spkm2kb09u70a3n5/branch/master?svg=true)](https://ci.appveyor.com/project/JFPerkins/allensdk-eye-tracking/branch/master)
[![codecov](https://codecov.io/gh/AllenInstitute/allensdk.eye_tracking/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenInstitute/allensdk.eye_tracking)

AllenSDK Eye Tracking
=====================

This is the python package the Allen Institute uses for estimating
pupil position and shape from eye videos. The position of a LED
reflection on the cornea is also tracked and is a required feature of
the input streams. The input videos are black and white.

Source: https://github.com/AllenInstitute/allensdk.eye_tracking

Installation
------------
The video IO is done using OpenCV's video functionality. Unfortunately,
OpenCV on pip seems to not be built with the necessary backend, as the
methods fail silently. As a result, we have not included OpenCV in the
requirements and it is necessary to get it seperately, built with the
video capture and writing functional. Additionally, on some platforms
scikit-image does not build easily from source and the developers don't
have bindary distributions for all platforms yet. The simplest way to
install these difficult dependencies is to use conda:

    conda install scikit-image
    conda install -c conda-forge opencv
    conda install -c conda-forge pyqt

The rest of the dependencies are all in the requirements, so to
install just clone or download the repository and then from inside the
top level directory either run:

    pip install .

or

    python setup.py install

Usage
-----
After installing the package, and entry point is created so it can be run
from the command line. To minimally run with the default settings:

    allensdk.eye_tracking --input_source <path to an input video>

To see all options that can be set at the command line:

    allensdk.eye_tracking --help

There are a lot of options that can be set, so often it can be more
convenient to store them in a JSON-formatted file which can be used like:

    allensdk.eye_tracking --input_json <path to the input json>

The input json can be combined with other command line argument, which will
take precedence over anything in the json. There is a UI tool for adjusting
and saving input parameters that can be used by running:

    allensdk.eye_tracking_ui

Description of algorithm
------------------------
The general way that the algorithm works is to (for every frame):

1. Use a simple bright circle template to estimate the seed point for
searching for a corneal reflection of the LED.
2. Draw rays from the seed point and find light-to-dark threshold
crossings to generate estimated points for an ellipse fit.
3. Use ransac to find the best fit ellipse to the points.
4. Optionally fill in the estimated corneal reflection with the last
shade of the pupil. This is necessary if the corneal reflection
occludes the pupil at all.
5. Repeat steps 1-3, but with a dark circle template and dark-to-light
threshold crossings to find the pupil ellipse parameters.
