from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_requirements = f.read().splitlines()

setup(
    name='aibs_eye_tracking',
    version='0.2.2',
    description="""AIBS package for mouse eye tracking.""",
    author="Jed Perkins",
    author_email="jedp@alleninstitute.org",
    url='https://github.com/AllenInstitute/aibs.eye_tracking',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'aibs.eye_tracking = aibs.eye_tracking.__main__:main'
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require=test_requirements
)
