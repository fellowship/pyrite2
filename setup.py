from setuptools import setup
from setuptools import find_packages

setup(
    name='pyrite',
    version='0.1.2',
    author='StartupML',
    author_email='team@startup.ml',
    license='LICENSE.txt',
    install_requires=['scikit-learn','numpy','scipy','pandas','astroML','wget','astroML_addons'],
    description='Anomaly Detection in High Dimensional Heterogeneous Datasets',
    packages=find_packages())
