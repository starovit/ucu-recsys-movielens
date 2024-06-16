Jupyter notebook with the experiment that evaluates baseline recommender (demonstrates mean-rating recommender with all metrics using different data splits (e.g., random, ordered, or user-based splitting).):
- **baseline_with_metrics.ipynb**.

Evaluation will be made mostly by ML metrics:
1. Predictive metrics like precision and recall at k are significantly affected by this sparsity of the dataset.
2. Rank metrics such as Mean Reciprocal Rank (MRR) and Hit Rate are sensitive to user bias in ratings.

## Baseline model
- A simple baseline model such as the mean-rating recommender. This model predicts the average rating of a movie based on historical ratings.

**Implementation**:
```python
class BaseModelAverage:
    def fit(self, train):
        self.rate_dict = train.groupby('MovieID')['Rating'].mean().to_dict()
        self.global_mean = train['Rating'].mean()

    def predict(self, movie_ids):
        return np.array([self.rate_dict.get(movie_id, self.global_mean) for movie_id in movie_ids])
```
**Usage**:
```python
from src.models import BaseModelAverage
```

## Metrics
**Metrics categories which were implemented:**
1. **Machine Learning Metrics**: MAE, RMSE, Precision, Recall, F1-Score, and ROC-AUC;
```python
def ml_metrics(true_scores, predicted_scores, threshold=4):
    .....
    return {
        "mae": round(mean_absolute_error(true_scores, predicted_scores), 3),
        "rmse": round(np.sqrt(mean_squared_error(true_scores, predicted_scores)), 3),
        "precision": round(precision_score(true_scores_cat, predicted_scores_cat), 3),
        "recall": round(recall_score(true_scores_cat, predicted_scores_cat), 3),
        "f1": round(f1_score(true_scores_cat, predicted_scores_cat), 3),
        "roc_auc": round(roc_auc_score(true_scores_cat, predicted_scores_cat), 3)
    }
```
2. **Ranking Metrics**: Mean Reciprocal Rank (MRR), Hit Rate. Measure how well the recommender ranks relevant items.
```python
def predictive_metrics(test: pd.DataFrame, predicted_scores, 
                       k=10, threshold=4):
    
    ...
    return {"k": k,
            "threshold": threshold,
            "precision_at_k": round(prec_at_k, 3),
            "recall_at_k": round(rec_at_k, 3),
            "avrg_prec_at_k": round(avrg_prec_at_k, 3),
            "n_users_with_k": n_users_with_k
            }
```
3. **Predictive Metrics**: Precision at K, Recall at K, and Average Precision at K.
Threshold is used to convert scores to categorical values. K can be speficied.
```python
def rank_metrics(test: pd.DataFrame, predicted_scores,  k=10, threshold=4):
    ...
    return {"mean_reciprocal_rank": round(mean_r_r, 3),
            "hit_rate": round(hit_r, 3)
            }
```

**Usage**:
```python
from src.metrics import ml_metrics, predictive_metrics, rank_metrics
```