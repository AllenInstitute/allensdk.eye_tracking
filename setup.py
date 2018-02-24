from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_requirements = f.read().splitlines()

setup(
    name='allensdk_eye_tracking',
    version='1.2.0',
    description="""Allen Institute package for mouse eye tracking.""",
    author="Jed Perkins",
    author_email="jedp@alleninstitute.org",
    url='https://github.com/AllenInstitute/allensdk.eye_tracking',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'allensdk.eye_tracking = allensdk.eye_tracking.__main__:main',
            'allensdk.eye_tracking_ui = allensdk.eye_tracking.ui.qt:main'
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require=test_requirements
)
