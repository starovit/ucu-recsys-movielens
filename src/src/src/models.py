from typing import Set, Any

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseModelAverage:
    """A simple baseline model that predicts movie ratings based on average ratings."""

    def fit(self, train: pd.DataFrame):
        self.rate_dict = train.groupby('MovieID')['Rating'].mean().to_dict()
        self.global_mean = train['Rating'].mean()

    def predict(self, movie_ids: list):
        """
        Predict ratings for a list of movie IDs.
        """
        prediction = [self.rate_dict.get(movie_id, self.global_mean) for movie_id in movie_ids]
        return np.array(prediction)


class ContentBasedModel:
    def __init__(self, num_of_movies_to_recommend=5):
        self._num_of_movies_to_recommend = num_of_movies_to_recommend
        self._movies_df = pd.DataFrame()

    def fit(self, df_movies):
        self._movies_df = df_movies.copy()
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Sci-Fi', 'SciFi')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Film-Noir', 'Noir')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Children\'s', 'Child')
        tfidf_vector = TfidfVectorizer(stop_words='english')
        # apply the object to the genres column
        tfidf_matrix = tfidf_vector.fit_transform(self._movies_df['Genres'])
        self._sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    def predict(self, movie_id):
        film_similarities = self._sim_matrix[movie_id]
        print(
            f"Let's try to find similar movies to film: {self._movies_df.iloc[movie_id]['Title']}, {self._movies_df.iloc[movie_id]['Genres']}")
        inxex_of_similarties = film_similarities.argsort()[::-1][:self._num_of_movies_to_recommend]
        for i in inxex_of_similarties:
            print(
                f"Movie with Id: {i}, Title: {self._movies_df.iloc[i]['Title']}, Genres: {self._movies_df.iloc[i]['Genres']} ")

        return inxex_of_similarties


class ItemItemModel:
    def __init__(self, rading_df: pd.DataFrame, num_of_similar_users=5):
        self._num_of_similar_users = num_of_similar_users
        self._ratings_df = rading_df.copy
        self._sim_matrix = {}
        self._user_movie_matrix = pd.DataFrame()

    def fit(self) -> None:
        self._user_movie_matrix = self._ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating',
                                                               fill_value=0)
        self._sim_matrix = cosine_similarity(self._user_movie_matrix)

    def predict(self, user_id) -> set[Any]:

        films_to_recommend, indexes_of_similar_users = self.find_films_that_other_users_like_but_current_user_havent_watched(user_id)
        filtered_recommendations = self.find_only_highly_rated_movies(films_to_recommend, indexes_of_similar_users)

        return filtered_recommendations

    def find_films_that_other_users_like_but_current_user_havent_watched(self, user_id, num_of_similar_users = 10):
        current_user = self._sim_matrix[user_id]
        indexes_of_similar_users = current_user.argsort()[::-1][:self._num_of_similar_users]

        films_to_recommend = set()
        for similar_user_index in indexes_of_similar_users:
            film_rates = self._user_movie_matrix.iloc[similar_user_index]
            for film_index, film_rate in film_rates.items():
                if self._user_movie_matrix.iloc[user_id][film_index] == 0 and film_rate != 0:
                    films_to_recommend.add(film_index)

        return films_to_recommend, indexes_of_similar_users

    def find_only_highly_rated_movies(self, films_to_check, indexes_of_similar_users, average_rating_threashold=4):
        filtered_recommendations = set()

        for film_index in films_to_check:
            all_film_rates = []
            for similar_user_index in indexes_of_similar_users:
                film = self._ratings_df[self._ratings_df['MovieID'] == film_index]
                if similar_user_index < len(film['Rating']):
                    rating = film['Rating'].iloc[similar_user_index]
                    all_film_rates.append(rating)

            average_rating = np.mean(all_film_rates)
            if average_rating >= average_rating_threashold:
                filtered_recommendations.add(film_index)

        return filtered_recommendations