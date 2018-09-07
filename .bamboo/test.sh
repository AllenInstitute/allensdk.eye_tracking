set -eu
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p ${bamboo_build_working_directory}/miniconda
export PATH=${bamboo_build_working_directory}/miniconda/bin:$PATH
export HOME=${bamboo_build_working_directory}/.home
export TMPDIR=${bamboo_build_working_directory}
export CONDA_PATH_BACKUP=${CONDA_PATH_BACKUP:-$PATH}
export CONDA_PREFIX=${CONDA_PREFIX:-}
export CONDA_PS1_BACKUP=${CONDA_PS1_BACKUP:-}
conda create -y -${bamboo_VERBOSITY} --prefix ${bamboo_build_working_directory}/.conda/${bamboo_TEST_ENVIRONMENT} python=3.6
source activate ${bamboo_build_working_directory}/.conda/${bamboo_TEST_ENVIRONMENT}
conda install -y -${bamboo_VERBOSITY} -c defaults scikit-image --clobber
conda install -y -${bamboo_VERBOSITY} -c conda-forge opencv=3.3.0 --clobber
cd ${bamboo_build_working_directory}/${bamboo_CHECKOUT_DIRECTORY}
pip install -r requirements.txt
pip install -r test_requirements.txt
pip install .
pytest --junitxml=test-reports/tests.xml
coverage run --source ./ -m pytest
coverage html --omit="test/*,setup.py"
source deactivate