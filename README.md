# Decision Tree Classifier Implementation and Evaluation

## Project Overview
This project benchmarks the performance of custom k-Nearest Nieghbors (kNN) against a Dummy Classifier baseline. Attention is placed on comparing statistical techniques for estimating model accuracy.

# Key Features
* **Dataset Splitting Strategies :** Implemented and compared Random Sub-Sampling, 10-Fold Cross Validation, and Bootstrap methods.
* **Mathematical Logic:** Performed entropy calculations and information gain to determine optimal splits.
* **Performance Analysis:** Comparative analysis of accuracy and error rates across different validation techniques to understand how dataset splitting impacts predictive robustness.

## Technology Stack
* **Language:** Python
* **Libraries:** Numpy

## To Run Unit Tests
Run `pytest --verbose`
* This command runs all the discovered tests in the project
* You can run individual test modules with
    * `pytest --verbose test_myclassifiers.py`
* Note: the `-s` flag can be helpful because it will show print statement output from test execution
