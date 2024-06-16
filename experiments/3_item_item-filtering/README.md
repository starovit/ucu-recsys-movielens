# Item-Item Collaborative Filtering

Item-Item Collaborative Filtering is a recommendation method that suggests items similar to a given item based on user interaction patterns. This model focuses on identifying items that have been similarly rated by users, leveraging the idea that similar items are often preferred by the same users.

## Model Overview

The `ItemItemModel` class is designed to recommend movies that are most similar to a specified movie based on user rating data.

### Key Features

- **Similarity Calculation**: Uses cosine similarity to compute the similarity between items based on user ratings.
- **Customizable Recommendations**: Allows the specification of the number of similar items to return.

### Implementation

Below is the complete implementation of the `ItemItemModel` class:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Set

class ItemItemModel:
    def __init__(self, num_of_similar_items=5):
        self._num_of_similar_items = num_of_similar_items
        self._sim_matrix = {}
        self._ratings_df = pd.DataFrame()
        self._user_movie_matrix = pd.DataFrame()

    def fit(self, rading_df: pd.DataFrame) -> None:
        """
        Fit the model using the ratings DataFrame by creating a user-movie rating matrix
        and computing the item-item similarity matrix.

        Parameters:
        rading_df (pd.DataFrame): DataFrame containing user ratings with 'UserID', 'MovieID', and 'Rating'.
        """
        self._ratings_df = rading_df.copy()
        self._user_movie_matrix = self._ratings_df.pivot_table(index='UserID',
                                                               columns='MovieID',
                                                               values='Rating',
                                                               fill_value=0)
        self._sim_matrix = cosine_similarity(self._user_movie_matrix.T)

    def predict(self, user_id, film_id, average_rating_threshold=4) -> set[Any]:
        """
        Predict similar items and their average rating above a given threshold.

        Parameters:
        user_id (int): The ID of the user.
        film_id (int): The ID of the film for which similar items are to be found.
        average_rating_threshold (float): The threshold above which ratings are considered.

        Returns:
        tuple: (mean rating of similar items, set of indices of similar items)
        """
        rates_that_users_given, indexes_of_similar_items = self.__find_films_that_other_users_like_but_current_user_havent_watched(user_id, film_id)
        rates_that_users_given = np.array(rates_that_users_given)
        filtered_rates = [rate for rate in rates_that_users_given if rate >= average_rating_threshold]
        return np.mean(filtered_rates), indexes_of_similar_items

    def __find_films_that_other_users_like_but_current_user_havent_watched(self, user_id, film_id):
        """
        Find films that are similar to the given film and that other users have rated.

        Parameters:
        user_id (int): The ID of the user.
        film_id (int): The ID of the film.

        Returns:
        tuple: (list of ratings, list of indices of similar items)
        """
        current_user = self._sim_matrix[film_id]
        indexes_of_similar_items = current_user.argsort()[::-1][:self._num_of_similar_items]

        rates_that_users_given = []
        for similar_item_index in indexes_of_similar_items:
            film_rate = self._user_movie_matrix.iloc[similar_item_index][film_id]
            if film_rate > 0:
                rates_that_users_given.append(film_rate)

        return rates_that_users_given, indexes_of_similar_items
```

### Usage

The `ItemItemModel` can be used to identify and display items similar to a specified item by providing its item ID. Below is a usage example demonstrating how to fit the model with a dataset and predict similar items for a specific user and item.

#### Example

```python
item_item_model = ItemItemModel(num_of_similar_items=300)

item_item_model.fit(rading_df=df_ratings)

# Predict similar items for a given user and movie ID (e.g., user ID 20, movie ID 100)
predicted_scores_for_each_film, indexes_of_similar_items = item_item_model.predict(user_id=20, film_id=100)
```

In this example, we use the first 10,000 items to make predictions faster.

```python
X_test = test[["UserID", "MovieID"]][:10000]
y_test = test["Rating"][:10000]

def predict_wrapper(row):
    user_id = row["UserID"]
    movie_id = row["MovieID"]
    predict, indexes_of_similar_items = item_item_model.predict(user_id, movie_id)
    return predict

predict = X_test.progress_apply(predict_wrapper, axis=1).tolist()

y_pred = pd.DataFrame(predict, columns=["Rating"])
y_pred.fillna(0, inplace=True)

print(f"RM metrics: {rm_metrics(y_test, y_pred)}")
print(f"Predictive metrics: {predictive_metrics(test, y_pred, k=5)}")
```
