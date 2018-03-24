set -eu
export PATH=/shared/utils.x86_64/anaconda2-4.3.1/bin:$PATH
export HOME=${bamboo_build_working_directory}/.home
export TMPDIR=${bamboo_build_working_directory}
export CONDA_PATH_BACKUP=${CONDA_PATH_BACKUP:-$PATH}
export CONDA_PREFIX=${CONDA_PREFIX:-}
export CONDA_PS1_BACKUP=${CONDA_PS1_BACKUP:-}
conda create -y -${bamboo_VERBOSITY} --clone ${bamboo_BASE_ENVIRONMENT} --prefix ${bamboo_build_working_directory}/.conda/${bamboo_TEST_ENVIRONMENT}
source activate ${bamboo_build_working_directory}/.conda/${bamboo_TEST_ENVIRONMENT}
conda install -y -${bamboo_VERBOSITY} -c scikit-image
conda install -y -${bamboo_VERBOSITY} -c conda-forge opencv
cd ${bamboo_build_working_directory}/${bamboo_CHECKOUT_DIRECTORY}
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -r test_requirements.txt
pip install .
pytest --junitxml=test-reports/tests.xml
coverage run --source ./ -m pytest
coverage html --omit="test/*,setup.py"
source deactivate