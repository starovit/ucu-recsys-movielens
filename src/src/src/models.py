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
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('|',' ')
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
    def __init__(self, num_of_similar_items=5):
        self._num_of_similar_items = num_of_similar_items
        self._sim_matrix = {}
        self._ratings_df = pd.DataFrame()
        self._user_movie_matrix = pd.DataFrame()

    def fit(self, rading_df: pd.DataFrame) -> None:
        # currently we have Movies on x axes and Users on Y axes. We should transpose this matrix
        self._ratings_df = rading_df.copy()
        self._user_movie_matrix = self._ratings_df.pivot_table(index='UserID',
                                                               columns='MovieID',
                                                               values='Rating',
                                                               fill_value=0)
        self._sim_matrix = cosine_similarity(self._user_movie_matrix)


    def predict(self, user_id, film_id, average_rating_threashold=4) -> set[Any]:

        rates_that_users_given, indexes_of_similar_items = self.__find_films_that_other_users_like_but_current_user_havent_watched(user_id, film_id)
        rates_that_users_given = np.array(rates_that_users_given)
        filtered_rates = [rate for rate in rates_that_users_given if rate >= average_rating_threashold]
        return np.mean(filtered_rates), indexes_of_similar_items

    def __find_films_that_other_users_like_but_current_user_havent_watched(self, user_id, film_id):
        current_user = self._sim_matrix[film_id]
        indexes_of_similar_items = current_user.argsort()[::-1][:self._num_of_similar_items]

        rates_that_users_given = []
        for similar_item_index in indexes_of_similar_items:
            film_rate = self._user_movie_matrix.iloc[similar_item_index][film_id]
            if film_rate > 0:
                rates_that_users_given.append(film_rate)

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
    

class UserUserModel:
    def __init__(self, df: pd.DataFrame) -> None:
        self.rating_matrix = df.pivot_table(index='UserID', columns='MovieID',\
                                  values='Rating', fill_value=0)
        self.neighbors_dict = {}


    def fit(self):
        """"
        Calculate user similarity and order all neighbours by similarity. Order is saving in neighbours_dict of instance 
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
        Params 
            df:                 DataFrame with users and movies
            n_closest_users:    how many closest users will be used for prediction
        """
        predict = []
        for _, row in df.iterrows():
            user = row["UserID"]
            movie = row["MovieID"]
            user_neighbors = self.neighbors_dict[user][:n_closest_users]
            filtered_df = self.rating_matrix[self.rating_matrix.index.isin(user_neighbors)]
            filtered_df = filtered_df.loc[:, movie]
            pred = filtered_df[filtered_df != 0].mean()
            predict.append(pred)
        return predict
    

    def predict_for_user(self, user_id: int, n_closest_users=30, n_movies=10):
        user_neighbors = self.neighbors_dict[user_id][:n_closest_users]
        filtered_df = self.rating_matrix[self.rating_matrix.index.isin(user_neighbors)]
        pred = filtered_df[filtered_df != 0].mean()
        movies = pred[pred.notna()].sort_values(ascending=False)
        return movies[:n_movies].index.values
