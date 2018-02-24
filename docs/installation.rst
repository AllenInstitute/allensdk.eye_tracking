============
Installation
============

The video IO is done using OpenCV's video functionality. Unfortunately,
OpenCV on pip seems to not be built with the necessary backend, as the
methods fail silently. As a result, we have not included OpenCV in the
requirements and it is necessary to get it seperately, built with the
video capture and writing functional. Additionally, on some platforms
scikit-image does not build easily from source and the developers don't
have bindary distributions for all platforms yet. The simplest way to
install these difficult dependencies is to use conda::

    conda install scikit-image
    conda install pyqt
    conda install -c conda-forge opencv

The rest of the dependencies are all in the requirements, so to install
just clone or download the repository and then from inside the top
level directory either run::

    pip install .

or::

    python setup.py install
