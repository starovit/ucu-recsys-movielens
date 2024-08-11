# Learning to Rank Model Experiment

This notebook focuses on implementing and evaluating learning-to-rank (LTR) models of MovieLens Dataset. The experiment explores the effectiveness of LTR models in optimizing the ranking of items based on relevance, particularly within the context of recommendation systems.

**Model Training:**
    - **XGBoost Model:** A learning-to-rank model is implemented using XGBoost, a robust and scalable algorithm for ranking tasks.
    - **Neural Network Model:** A custom neural network model is built to learn to rank items based on user-item interactions.

**Evaluation:**
    - The performance of both models is evaluated using ranking-specific metrics such as NDCG (Normalized Discounted Cumulative Gain), MAP (Mean Average Precision), and other relevant metrics.
    - Visualizations and analysis are provided to interpret the results and assess the models' effectiveness.

## Neural Network Model

The neural network model used in this experiment is defined by the following classes:

### MovieLensDataset

The `MovieLensDataset` class handles the loading and preparation of data for training the neural network. It creates tensors for user IDs, movie IDs, and corresponding ratings.

```python
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(data['movie_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['Rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
```

### RankingNetwork

The `RankingNetwork` class defines a neural network architecture for the ranking task. It utilizes embedding layers for users and movies, followed by fully connected layers to predict the relevance score for each user-movie pair.

```python
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
```

### Predicting Top Movies

The model includes a method to predict the top-rated movies for a given user:

```python
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
```

### Model Evaluation

The `evaluate` method assesses the model's performance using the evaluation metrics MAP and NDCG:

```python
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
```

## Pros and Cons

### Pros

- **Efficiency:** The models can handle large datasets, making them suitable for real-world ranking tasks.
- **Scalability:** Both the XGBoost and neural network models are scalable to large data volumes.
- **Relevance:** The models improve the relevance of ranked items, tailoring recommendations based on learned patterns.

### Cons

- **Complexity:** The process of training and tuning the models can be complex, especially with the neural network model, requiring careful selection of features and parameters.
- **Data Dependency:** The effectiveness of the models is highly dependent on the quality and quantity of the data available.

