### Pyrite Overview

Pyrite from [Startup.ML](http://startup.ml) is an anomaly detection tool optimized for high-dimensional, heterogeneous (both categorical and continuous) datasets.   Its time complexity is near linear and space complexity is quadratic to the number of dimensions.

Pyrite is designed for categorical features (e.g., city can contain 'San Francisco', 'Boston', etc.) however it can also handle numerical features (e.g., income, age, etc) through discretization.    

### Getting started

```
from pyrite import Pyrite
pyrite_data = Pyrite(pandas.read_csv('expanded.csv'))
score_vec = pyrite_data.score_dataset(50, 100)
```

### Installation

Pyrite uses the following dependencies:

- __numpy__
- __pandas__
- __matplotlib__

Once you have the dependencies installed, [download the distro](https://lendingai.s3.amazonaws.com/assets/dist/pyrite-0.1.2.tar.gz):
```bash
wget https://lendingai.s3.amazonaws.com/assets/dist/pyrite-0.1.2.tar.gz
tar xvfz pyrite-0.1.2.tar.gz
```

Go to the Pyrite folder and run the install command:
```bash
cd pyrite
sudo python setup.py install
```
Please review the [license terms](license.md) before installing and using Pyrite.

### Support

Please email your questions and comments to [Startup.ML](http://startup.ml/connect).
