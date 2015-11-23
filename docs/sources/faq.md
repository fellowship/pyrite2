# Frequently Asked Questions

### How accurate is Pyrite?
Pyrite is an unsupervised anomaly detection tool so its rate of detection will be dependent on input feature characteristics, and to a lesser extent, parameter selection.
When applied to well known datasets, it's able to identify minority class members (obviously without the use of the label) at a rate of 60%-90%.

### What does the anomaly score mean?
The anomaly score is essentially an indicator for how rare this particular observation is in the context of the entire dataset. The anomaly score is scaled between 0 and 1. 

### Is Pyrite a parametric or non-parametric method?
Pyrite is a non-parametric method. Whenever it scores an observation, it is comparing it to the entire dataset (it uses certain sampling methods internally to make this tractable).

### How fast is Pyrite?
Pyrite's time complexity is near linear and space complexity is quadratic to the number of dimensions. In most datasets, anomalies occur as a combination of multiple features. 
Pyrite is able to detect such anomalies without requiring O(N^2) time complexity. As an example, a dataset with 10,000 observations and 20 features will take ~3 seconds to process.

### Is there a UI for Pyrite?
Pyrite is a python module. It can be used from another python program, from the command line or in iPython Notebook.

### Where can I get the source code?
Pyrite is not open source.  Please review the [license terms](license.md).

### Where can I get help?
Please email your questions and comments to [Startup.ML](http://startup.ml/connect).
