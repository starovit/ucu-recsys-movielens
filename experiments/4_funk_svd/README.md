# Funk SVD Algorithm Implementation

This repository contains the implementation of Simon Funk's famous Singular Value Decomposition (SVD) algorithm for collaborative filtering, primarily used in recommendation systems.

## Algorithm Steps

1. **Initialization**:
    - Initialize user and item biases (`bu`, `bi`).
    - Initialize user and item latent factors matrices (`pu`, `qi`).

2. **Training**:
    - For each epoch:
        - Shuffle the training data (if specified).
        - Run Stochastic Gradient Descent (SGD) to update biases and latent factors.
        - If a validation set is provided, compute validation metrics (Loss, RMSE, MAE).
        - Check for early stopping based on validation RMSE.

3. **Prediction**:
    - Compute the estimated rating for a given user-item pair using the learned biases and latent factors.
    - Optionally clip the predicted rating to a specified range.


## Loading the Model

The model can be loaded by initializing the `SVD` class with the desired parameters and calling the `fit` method with the training and optionally validation data.

### Example Usage

```python
import pandas as pd
from svd import SVD

# Load your data
train_data = pd.DataFrame({
    'u_id': [1, 2, 3, 4, 5],
    'i_id': [1, 2, 3, 4, 5],
    'rating': [5, 4, 3, 2, 1]
})

val_data = pd.DataFrame({
    'u_id': [1, 2, 3],
    'i_id': [1, 2, 3],
    'rating': [5, 4, 3]
})

# Initialize the SVD model
svd = SVD(lr=0.005, reg=0.02, n_epochs=20, n_factors=100, early_stopping=True, shuffle=True)

# Fit the model
svd.fit(train_data, val_data)

# Make predictions
predictions = svd.predict(val_data)

print(predictions)
```

## Class Documentation

### `SVD`

#### Parameters:
- `lr` (float): Learning rate. Default is 0.005.
- `reg` (float): L2 regularization factor. Default is 0.02.
- `n_epochs` (int): Number of SGD iterations. Default is 20.
- `n_factors` (int): Number of latent factors. Default is 100.
- `early_stopping` (bool): Whether or not to stop training based on a validation monitoring. Default is False.
- `shuffle` (bool): Whether or not to shuffle the training set before each epoch. Default is False.
- `min_delta` (float): Minimum delta to argue for an improvement. Default is 0.001.
- `min_rating` (int): Minimum value a rating should be clipped to at inference time. Default is 1.
- `max_rating` (int): Maximum value a rating should be clipped to at inference time. Default is 5.

#### Methods:
- `fit(X, X_val=None)`: Learns model weights from input data.
- `predict(X, clip=True)`: Returns estimated ratings of several given user/item pairs.

## Dependencies

- `numpy`
- `pandas`
- `funk-svd`
