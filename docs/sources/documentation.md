# Sibyl: Anomaly Detection in highly dimensional datasets

## Introduction

The Sibyl module, provides a way to discover anomalies in your categorical datasets. If your data contains numberical attributes mixed with categorical ones, the method can be extended to take care of that case as well. But first, any numerical attributes need to be discretized before being supplied to the Sibyl function. Sibyl also calculates the ***features*** and ***feature pairs*** that were the main contributers in the anomaly score of some instance y.


## Methods Descriptions
```python
sibyl(dataset, ssamples_num, ssample_size, anomalies)
```

The main method of the sibyl module, and it takes the following parameters as an input:
