## Anomaly Scoring

Sibyl can be used to compute an anomaly score for each instance in the dataset, or for a single instance.

- __Methods__:
  -__score_dataset__(sample_num, sample_size)
    - __Return__: numpy.ndarray (a list where the i_th element is the anomaly score for the i_th instance in the dataset).
    - __Arguments__:
      - __sample_num__: int (number of random samples chosen without replacement from the dataset).
      - __sample_size__: int (size of the random sample)

  -__score_instance__(single_instance, sample_num, sample_size)
    - __Return__: float (Anomaly score for the single instance)
    - __Arguments__:
      - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
      - __sample_num__: int (number of random samples chosen without replacement from the dataset)
      - __sample_size__: int (size of the random sample)

