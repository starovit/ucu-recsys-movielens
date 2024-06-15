import numpy as np
import pandas as pd

class BaseModelAverage:
    """A simple baseline model that predicts movie ratings based on average ratings."""
    def fit(self, train: pd.DataFrame):
        self.rate_dict = train.groupby('MovieID')['Rating'].mean().to_dict()
        self.global_mean = train['Rating'].mean()

    def predict(self, movie_ids: list):
        """
        Predict ratings for a list of movie IDs.
        """
        prediction =  [self.rate_dict.get(movie_id, self.global_mean) for movie_id in movie_ids]
        return np.array(prediction)
