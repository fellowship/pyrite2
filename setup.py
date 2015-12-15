from setuptools import setup
from setuptools import find_packages

setup(
    name='pyrite2',
    version='0.1.7',
    author='StartupML',
    author_email='team@startup.ml',
    license='LICENSE.txt',
    url='http://pyrite.startup.ml',
    install_requires=['scikit-learn','numpy','scipy','pandas','astroML','wget','astroML_addons','matplotlib'],
    description='Anomaly Detection in High Dimensional Heterogeneous Datasets',
    packages=find_packages())
