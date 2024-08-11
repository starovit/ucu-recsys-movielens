# Two-Stage Recommendation System Pipeline

## Overview
This project implements a two-stage recommendation system pipeline using collaborative filtering for candidate generation and XGBoost for ranking. The pipeline is designed to handle large datasets typical in recommendation system environments, focusing on movies rating predictions.

### Pipeline Structure
1. **Candidate Generation**: The first stage uses user-user collaborative filtering to identify a set of candidate movies for each user. This is achieved by computing similarities between users based on their ratings and selecting the top movies rated by the most similar users.
2. **Ranking**: The second stage ranks these candidates using an XGBoost model trained to predict movie ratings based on user and movie features. This stage fine-tunes the recommendations by learning complex patterns in the data.

## Implementation Details
- **Data Handling**: Uses `pandas` and `numpy` for data manipulation and preparation.
- **Model Training and Prediction**: Utilizes `xgboost` for creating and using the ranking model.
- **Metrics Calculation**: Employs metrics like Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG) to evaluate the quality of the recommendations.

### Code Structure
- Data is loaded and preprocessed to handle missing values and ensure consistency.
- A user similarity matrix is constructed for the candidate generation phase.
- Ratings are predicted using a pre-trained XGBoost model.
- Final predictions are evaluated using ranking-specific performance metrics.

## Pros and Cons

### Pros
- **Scalability**: Efficiently narrows down the vast number of potential recommendations by filtering through the candidate generation stage before ranking, making it scalable to large datasets.
- **Relevance**: Increases the relevance of the recommendations by focusing the ranking stage on items already pre-filtered as potentially interesting to the user.
- **Customizability**: Both stages can be independently adjusted or replaced with different algorithms or techniques to experiment with or improve performance.

### Cons
- **Complexity**: The two-stage process adds complexity to the system, requiring careful tuning and integration of the two parts.
- **Error Propagation**: Errors in the candidate generation phase can significantly impact the final recommendations, as no amount of sophisticated ranking can recover from poor initial candidate selection.
- **Cold Start Problem**: Struggles with new users or items that have little to no historical data, which might be mitigated but not entirely solved by the current approach.