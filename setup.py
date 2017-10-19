from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt','r') as f:
    test_requirements = f.read().splitlines()

setup(
    name = 'aibs_eye_tracking',
    version = '0.1.0',
    description = """AIBS package for rodent eye tracking.""",
    author = "Jed Perkins",
    author_email = "jedp@alleninstitute.org",
    url = 'http://stash.corp.alleninstitute.org/scm/~jedp/aibs.eye_tracking',
    packages = find_packages(),
    include_package_data=True,
    install_requires = requirements,
    entry_points={
          'console_scripts': [
              'aibs.eye_tracking = aibs.eye_tracking.__main__:main'
        ]
    },
    license="Allen Institute Software License",
    setup_requires=['pytest-runner'],
    tests_require = test_requirements
)
