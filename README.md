# Anime Recommendation System

A machine learning-based recommendation system for anime that predicts personalized user ratings and generates recommendations based on user preferences and anime characteristics.

## Project Overview

This project implements a robust anime recommendation system using machine learning techniques to provide personalized anime recommendations. The system analyzes user ratings, anime features, genre preferences, and viewing behaviors to predict how users would rate anime they haven't watched yet.

## Features

- **Personalized Recommendations**: Predicts ratings for unwatched anime based on user preferences
- **Comprehensive Feature Engineering**: Utilizes both user-based and anime-based features
- **Multiple ML Models**: Implements Decision Tree and Random Forest regression models
- **Performance Metrics**: Evaluates using industry-standard metrics (RMSE, MAE)
- **Support for New Users**: Handles cold-start problem for new users
- **Data Visualization**: Provides insights through visualizations

## Dataset

The system uses the Anime Recommendations Database containing:

- **Anime Information**: Details about 12,000+ anime including genres, type, episodes, ratings
- **User Ratings**: 7 million+ ratings from 73,000+ users

## Directory Structure

```
├── data/                 # Data directory with raw and processed datasets
├── models/               # Trained machine learning models
├── src/                  # Source code
├── overview.ipynb        # Jupyter notebook with project overview and visualization
├── improved_anime_visualizations.py  # Visualization scripts
├── data_downloading.py   # Utility to download and prepare the dataset
└── anime_dataset_info.txt  # Dataset documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anime-recommendation-system.git

# Navigate to project directory
cd anime-recommendation-system

# Install required packages
pip install -r requirements.txt

# Download and prepare the dataset
python data_downloading.py
```

## Usage

### Generating Recommendations

```python
from models.anime_recommender import AnimeRecommender

# Initialize the recommender
recommender = AnimeRecommender()

# Load a pre-trained model
recommender.load_model('models/random_forest_model.pkl')

# Get recommendations for an existing user
recommendations = recommender.recommend_for_user(user_id=42, n=10)

# Get recommendations for a new user based on ratings
new_user_ratings = {
    'Death Note': 9,
    'Attack on Titan': 10,
    'One Punch Man': 8
}
new_user_recommendations = recommender.recommend_for_new_user(new_user_ratings, n=10)
```

## Model Performance

The following table summarizes the performance of various models evaluated:

![Model Performance Results](experiment%20results.png)

## Visualizations

The project includes comprehensive visualizations for:
- User rating distributions
- Genre popularity analysis
- Feature importance
- Recommendation quality metrics

## Contributor:
Nate Hu, Stuart Lin, Junhao Huang