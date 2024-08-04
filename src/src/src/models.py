from typing import Set, Any

import numpy as np
import pandas as pd
from scipy import stats


from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score
from src.metrics import ml_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm


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

        return rates_that_users_given, indexes_of_similar_items

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
    


class MovieDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop('Rating', axis=1).values
        self.labels = dataframe['Rating'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class MovieRatingNN(nn.Module):
    def __init__(self, num_features):
        super(MovieRatingNN, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output_reg = nn.Linear(64, 1)

    def forward(self, x):
        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = self.output_reg(x)
        x = torch.sigmoid(x) * 4 + 1
        return x

class IncrementalABTester:
    def __init__(self):
        self.true_ratings = np.array([])
        self.predictions_baseline = np.array([])
        self.predictions_nn = np.array([])

    def add_data(self, true_rating, prediction_baseline, prediction_nn):
        self.true_ratings = np.append(self.true_ratings, true_rating)
        self.predictions_baseline = np.append(self.predictions_baseline, prediction_baseline)
        self.predictions_nn = np.append(self.predictions_nn, prediction_nn)

    def rmse(self, predictions):
        return np.sqrt(np.mean((self.true_ratings - predictions) ** 2))

    def run_tests(self):
        rmse_baseline = self.rmse(self.predictions_baseline)
        rmse_nn = self.rmse(self.predictions_nn)
        
        # Conduct a paired t-test if we have enough data
        if len(self.true_ratings) > 1:
            _, p_value = stats.ttest_rel(self.predictions_baseline, self.predictions_nn)
        else:
            p_value = np.nan  # Not enough data to test
        
        results = {
            'rmse_baseline': rmse_baseline,
            'rmse_nn': rmse_nn,
            'p_value': p_value
        }
        
        return results
    
    

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(data['movie_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['Rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]



class RankingNetwork(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(RankingNetwork, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_all_movies(self, user_id, num_top_movies=10):
        self.eval()
        with torch.no_grad():
            all_movie_ids = torch.arange(self.movie_embedding.num_embeddings)
            user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long)
            predicted_ratings = self.forward(user_tensor, all_movie_ids).squeeze()
            _, top_indices = torch.topk(predicted_ratings, num_top_movies)
            top_movie_ids = all_movie_ids[top_indices].tolist()
            top_ratings = predicted_ratings[top_indices].tolist()
        
        results_df = pd.DataFrame({
            'MovieID': top_movie_ids,
            'PredictedRating': top_ratings
        })
        return results_df
    
    def evaluate(self, data_loader):
        self.eval()
        all_true_scores = []
        all_predicted_scores = []

        with torch.no_grad():
            for users, movies, ratings in data_loader:
                outputs = self(users, movies).squeeze()
                all_predicted_scores.extend(outputs.numpy())
                all_true_scores.extend(ratings.numpy())

        true_scores = np.array(all_true_scores)
        predicted_scores = np.array(all_predicted_scores)

        metrics = ml_metrics(true_scores, predicted_scores)

        true_relevance = (true_scores >= 4).astype(int) 
        sorted_indices = np.argsort(-predicted_scores) 
        sorted_true_relevance = true_relevance[sorted_indices]

        metrics['map'] = label_ranking_average_precision_score([sorted_true_relevance], [sorted_indices])
        metrics['ndcg'] = ndcg_score([sorted_true_relevance], [predicted_scores[sorted_indices]])

        return metrics