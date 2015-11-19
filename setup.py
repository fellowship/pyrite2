from setuptools import setup
from setuptools import find_packages

setup(
    name='sibyl',
    version='0.1.0',
    author='StartupML',
    author_email='team@startup.ml',
    license='LICENSE.txt',
    install_requires=['pandas'],
    description='Anomaly Detection in High Dimensional Heterogeneous Datasets',
    packages=find_packages())
