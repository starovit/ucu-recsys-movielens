# Content-Based Filtering

Content-Based Filtering is a recommendation approach that suggests items similar to a given item based on their content features.

## Model Overview

The `ContentBasedModel` class recommends movies that are most similar to a specified movie based on its genre. The model uses TF-IDF (Term Frequency-Inverse Document Frequency) to transform genre data into numerical features and calculates cosine similarity to find similar movies.

### Key Features

- **Similarity Calculation**: Converts genre text into numerical features using TF-IDF and computes similarity using cosine similarity.
- **Customizable Recommendations**: Allows customization of the number of similar movies to return.

### Implementation

Below is the complete implementation of the `ContentBasedModel` class:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedModel:
    def __init__(self, num_of_movies_to_recommend=5):
        self._num_of_movies_to_recommend = num_of_movies_to_recommend
        self._movies_df = pd.DataFrame()
    
    def fit(self, df_movies):
        """
        Fit the model using the movies DataFrame. This involves cleaning genre names
        and computing the TF-IDF matrix for the genres.

        Parameters:
        df_movies (pd.DataFrame): DataFrame containing movie information including 'Genres'.
        """
        self._movies_df = df_movies.copy()
        
        # Replace inconsistent genre names
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Sci-Fi', 'SciFi')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace('Film-Noir', 'Noir')
        self._movies_df['Genres'] = self._movies_df['Genres'].str.replace("Children's", 'Child')
        
        # Initialize the TF-IDF vectorizer and fit-transform the genres column
        tfidf_vector = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vector.fit_transform(self._movies_df['Genres'])
        
        # Compute the similarity matrix
        self._sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    def predict(self, movie_id):
        """
        Predict and print the most similar movies to the given movie ID.

        Parameters:
        movie_id (int): The ID of the movie for which similar movies are to be found.

        Returns:
        list: Indices of the most similar movies.
        """
        film_similarities = self._sim_matrix[movie_id]
        
        print(f"Finding movies similar to: {self._movies_df.iloc[movie_id]['Title']} "
              f"({self._movies_df.iloc[movie_id]['Genres']})")
        
        index_of_similarities = film_similarities.argsort()[::-1][:self._num_of_movies_to_recommend]
        
        for i in index_of_similarities:
            print(f"Movie ID: {i}, Title: {self._movies_df.iloc[i]['Title']}, Genres: {self._movies_df.iloc[i]['Genres']}")
        
        return index_of_similarities
```

### Usage

The `ContentBasedModel` can be used to identify and display movies similar to a specified movie by providing its movie ID. Below is an example of fitting the model with a dataset and predicting similar movies:

```python
model = ContentBasedModel(num_of_movies_to_recommend=15)

model.fit(df_movies=df_movies)

model.predict(movie_id=100)
```
