### Sibyl Overview

Sibyl from [Startup.ML](http://startup.ml) is an anomaly detection tool optimized for high-dimensional, heterogeneous (both categorical and continuous) datasets.   Its time complexity is near linear and space complexity is quadratic to the number of dimensions.

Sibyl is designed for categorical features (e.g., city can contain 'San Francisco', 'Boston', etc.) however it can also handle numerical features (e.g., income, age, etc) through discretization.    

### Getting started

```
from sibyl import Sibyl
sibyl_data = Sibyl(pandas.read_csv('expanded.csv'))
score_vec = sibyl_data.score_dataset(50, 100)
```

### Installation

Sibyl uses the following dependencies:

- __numpy__
- __pandas__
- __matplotlib__

Once you have the dependencies installed, download the distro:
```bash
wget http://...
```
Go to the Sibyl folder and run the install command:
```bash
cd sibyl
sudo python setup.py install
```
Please review the [license terms](license.md) before installing and using Sibyl.

### Support

Please email your questions and comments to [Startup.ML](http://startup.ml/connect).
