# User-User Collaborative Filtering

User-User Collaborative Filtering is a recommendation method that suggests items to a user based on the preferences of similar users. This model identifies users with similar taste and uses their ratings to predict what a given user might like.

## Model Overview

The `UserUserModel` class recommends movies to a user based on the ratings of their closest neighbors (users with similar rating patterns). The model calculates the cosine similarity between users and uses this similarity to identify and predict the preferences of a target user.

### Key Features

- **User Similarity Calculation**: Uses cosine similarity to measure the similarity between users based on their ratings.
- **Customizable Neighborhood**: Allows the specification of how many closest neighbors to use for predictions.

### Implementation

Below is the complete implementation of the `UserUserModel` class:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserUserModel:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the model by creating a user-movie rating matrix and preparing a dictionary
        for storing ordered neighbors based on similarity.

        Parameters:
        df (pd.DataFrame): DataFrame containing user ratings with 'UserID', 'MovieID', and 'Rating'.
        """
        self.rating_matrix = df.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)
        self.neighbors_dict = {}

    def fit(self):
        """
        Calculate user similarity and order all neighbors by similarity. The order is stored in the
        neighbors_dict attribute of the instance.

        Returns:
        self
        """
        user_similarity = pd.DataFrame(cosine_similarity(self.rating_matrix), index=self.rating_matrix.index,
                                       columns=self.rating_matrix.index)
        for i in range(user_similarity.shape[0]):
            row = user_similarity.iloc[i]
            user = row.index[i]
            row = row[row.index != user]
            neighbors = list(np.argsort(row)[::-1])
            self.neighbors_dict[user] = neighbors
        return self
    
    def predict(self, df: pd.DataFrame, n_closest_users: int):
        """
        Predict ratings for the specified user-movie pairs using the n closest neighbors.

        Parameters:
        df (pd.DataFrame): DataFrame with 'UserID' and 'MovieID' columns specifying the pairs to predict.
        n_closest_users (int): The number of closest users to use for prediction.

        Returns:
        list: Predicted ratings for the specified user-movie pairs.
        """
        predictions = []
        for _, row in df.iterrows():
            user = row["UserID"]
            movie = row["MovieID"]
            user_neighbors = self.neighbors_dict[user][:n_closest_users]
            filtered_df = self.rating_matrix[self.rating_matrix.index.isin(user_neighbors)]
            filtered_df = filtered_df.loc[:, movie]
            pred = filtered_df[filtered_df != 0].mean()
            predictions.append(pred)
        return predictions
    
    def predict_for_user(self, user_id: int, n_closest_users=30, n_movies=10):
        """
        Predict top movies for a specific user using the n closest neighbors.

        Parameters:
        user_id (int): The ID of the user for whom recommendations are to be made.
        n_closest_users (int, optional): The number of closest users to use for prediction. Default is 30.
        n_movies (int, optional): The number of top movies to recommend. Default is 10.

        Returns:
        np.array: Array of movie IDs recommended for the user.
        """
        user_neighbors = self.neighbors_dict[user_id][:n_closest_users]
        filtered_df = self.rating_matrix[self.rating_matrix.index.isin(user_neighbors)]
        pred = filtered_df[filtered_df != 0].mean()
        movies = pred[pred.notna()].sort_values(ascending=False)
        return movies[:n_movies].index.values
```

### Usage

The `UserUserModel` can be used to recommend movies to a user based on their similarity to other users. Below is an example demonstrating how to fit the model with a dataset and predict ratings for a set of user-movie pairs, as well as recommend movies for a specific user.

#### Example

```python
# Initialize the model with the user ratings DataFrame
user_user_model = UserUserModel(df=df_ratings)

# Fit the model to calculate user similarities
user_user_model.fit()

# Predict ratings for a given user-movie pairs DataFrame
predictions = user_user_model.predict(df=test[["UserID", "MovieID"]], n_closest_users=30)

# Predict top movies for a specific user (e.g., user ID 42)
recommended_movies = user_user_model.predict_for_user(user_id=42, n_closest_users=30, n_movies=10)
```

### Important Notes

- **User Similarity Calculation**: The model calculates user similarities based on their ratings to identify the most similar users, allowing for personalized recommendations.
- **Neighborhood Size**: The number of closest users used for predictions can be adjusted to balance between recommendation accuracy and computational efficiency.

