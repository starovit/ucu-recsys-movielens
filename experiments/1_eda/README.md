# Exploratory Data Analysis of the MovieLens Dataset

This repository contains an exploratory data analysis (EDA) of the MovieLens 1M dataset.

## Conclusions

1) The dataset is well-prepared and free of missing (None) values.
2) The most common genres are drama, comedy, and action, while noir, fantasy, and western are the least represented. 
3) Approximately 25% of the films were made between 1997 and 2000.
4) The dataset is gender unbalanced with 70% male and 30% female users. The gender and age imbalances should be considered when designing models to avoid biased recommendations.
5) The second most common occupation category is "not specified," with "student" being the most common.
6) The typical user is between 18 and 24 years old.
7) Most of the users are from the state of California, which is expected given it's the most populated state.
8) There are outlier days when more than 50,000 reviews were written, suggesting possible artificial data.
9) Only 83% of the movies have at least 10 reviews. The highest-rated movie is "Sanjuro," while "American Beauty" is the most popular.
10) The average movie rating is 3.3, whereas the average rating given by users is 3.7.
11) Users write an average of 165 reviews, but the mode is only 21.
12) Older/retired individuals and females tend to give higher ratings, whereas the lowest ratings come from younger, unemployed individuals.
13) War movies receive the highest ratings, while horror movies receive the lowest.This info can help to make genre-aware recommendation algorithms.
14) Ratings correlate with the year of the movie release, peaking between 1940-1960 and then declining.
- Outliers should be checked and deleted (such as user who put rating 1 for 100 movies)

## Data Splitting for Training and Testing
- Splitting Date: "2000-12-02"
- Train: 80% of records, including 5390 users (90% of unique users).
- Test: 20% of records, including 1802 users (30% of unique users).