from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
        packages=['open3d_conversions'],
        scripts=[],
        package_dir={'': 'src'}
        )

setup(**d)

