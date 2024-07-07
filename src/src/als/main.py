import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score

import numpy as np
from tqdm import tqdm



class AlternatingLeastSquares:
    def __init__(self, num_factors=10, regularization=0.1, iterations=10):
        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, interaction_matrix):
        self.num_users, self.num_items = interaction_matrix.shape
        self.user_factors = np.random.random((self.num_users, self.num_factors))
        self.item_factors = np.random.random((self.num_items, self.num_factors))

        for iteration in tqdm(range(self.iterations), desc="ALS Training Progress"):
            self.user_factors = self._als_step(interaction_matrix, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(interaction_matrix.T, self.item_factors, self.user_factors)


    def _als_step(self, interaction_matrix, update_vecs, fixed_vecs):
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.num_factors) * self.regularization
        b = interaction_matrix.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        update_vecs = b.dot(A_inv)
        return update_vecs


    def predict(self, user_id=None):
        if user_id is None:
            predictions = self.user_factors.dot(self.item_factors.T)
            return predictions
        else:
            predictions =  self.user_factors.dot(self.item_factors.T)
            print(predictions.shape)
            return predictions[user_id]

    def calculate_mse(self, interaction_matrix):
        if isinstance(interaction_matrix, pd.DataFrame):
            interaction_matrix = interaction_matrix.values

        predictions = self.predict()
        mse = mean_squared_error(interaction_matrix, predictions)
        return mse

    def calculate_f1(self, interaction_matrix, threshold=0.5):
        if isinstance(interaction_matrix, pd.DataFrame):
            print("dataframe detected")
            interaction_matrix = interaction_matrix.values

        predictions = self.predict()
        binary_predictions = (predictions >= threshold).astype(int)
        binary_interactions = (interaction_matrix >= threshold).astype(int)
        f1 = f1_score(np.array(binary_interactions).flatten(), np.array(binary_predictions).flatten())
        return f1
