# Incremental OnlineA/B Tester README

## Overview
`IncrementalABTester` facilitates A/B testing between two predictive models using time-sorted data. It computes MSE and conducts paired t-tests to evaluate model performance incrementally.

## Functionality
- **add_data**: Inputs true rating, baseline prediction, and neural network prediction.
- **mse**: Calculates Mean Square Error.
- **run_tests**: Computes RMSE for each model and conducts a paired t-test if enough data is present.

## Statistical Method
Utilizes paired t-tests to determine if differences in model performance are statistically significant, requiring a minimum of two data points.

## Usage Example
```python

# Initialize tester
tester = IncrementalABTester()

# Add data points
tester.add_data(4, 3.5, 4.1) # true # (group A) # (group B)

# Evaluate models
results = tester.run_tests()
print(results)
```