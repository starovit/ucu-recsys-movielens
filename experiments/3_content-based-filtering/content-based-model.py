import pandas as pd

from src.utils import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedModel:
    def __init__(self, num_of_movies_to_recommend=5):
        self._num_of_movies_to_recommend = num_of_movies_to_recommend
        self._movies_df = pd.DataFrame()

    def fit(self, df_movies):
        self._movies_df = df_movies.copy()
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Sci-Fi','SciFi')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Film-Noir','Noir')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Children\'s','Child')
        tfidf_vector = TfidfVectorizer(stop_words='english')
        # apply the object to the genres column
        tfidf_matrix = tfidf_vector.fit_transform(self._movies_df['Genres'])
        self._sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    def predict(self, movie_id):
        film_similarities = self._sim_matrix[movie_id]
        print(f"Let's try to find similar movies to film: {self._movies_df.iloc[movie_id]['Title']}, {self._movies_df.iloc[movie_id]['Genres']}")
        inxex_of_similarties = film_similarities.argsort()[::-1][:self._num_of_movies_to_recommend]
        for i in inxex_of_similarties:
            print(f"Movie with Id: {i}, Title: {self._movies_df.iloc[i]['Title']}, Genres: {self._movies_df.iloc[i]['Genres']} ")

        return inxex_of_similarties
