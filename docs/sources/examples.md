## Examples

```python
from sibyl import Sibyl
data = pandas.read_csv('expanded.csv', header=None)

# Return anomaly score for every sample in the dataset
sibyl_data = Sibyl(data)
score_vec = sibyl_data.score_dataset(50, 100)

# Return anomaly score for a single random instance from the dataset
rand_instance = data.iloc[30]
score_30 = sibyl_data.score_instance(rand_instance, 50, 100)
```
