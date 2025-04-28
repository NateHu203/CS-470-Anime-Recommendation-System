import pandas as pd
import numpy as np
import os
import logging
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="anime_adaboost_recommender.log"
)
logger = logging.getLogger(__name__)

class AnimeAdaBoostRecommender:
    """Anime recommendation system using AdaBoost regressor."""
    
    def __init__(self, data_path='data/processed/'):
        """Initialize the recommendation system with data path."""
        self.data_path = data_path
        self.ada_model = None
        self.anime_features = None
        self.feature_columns = None
        self.anime_dict = {}  # For quick lookups
        
    def load_data(self):
        """Load the necessary data files for the recommendation system."""
        print("Loading data files...")
        
        try:
            # Load the preprocessed data files
            self.anime_features = pd.read_csv(os.path.join("..", self.data_path, 'anime_features_normalized_new.csv'))
            self.rating_data = pd.read_csv(os.path.join("..", self.data_path, 'rating_enhanced_new.csv'))
            
            # Load train and test sets
            train_data = pd.read_csv(os.path.join("..", self.data_path, 'rating_train_v2_new.csv'))
            test_data = pd.read_csv(os.path.join("..", self.data_path, 'rating_test_v2_new.csv'))
            
            # Create a dictionary for quick anime lookups
            self.anime_dict = self.anime_features.set_index('anime_id').to_dict('index')
            
            print(f"Data loaded successfully. Training set: {len(train_data)} samples, Test set: {len(test_data)} samples")
            logger.info(f"Data loaded. Train: {len(train_data)}, Test: {len(test_data)}")
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, train_df, test_df):
        """Prepare features for model training from the dataset."""
        print("Preparing features for training...")
        
        # Create a copy of anime_features to avoid modifying the original
        anime_features_copy = self.anime_features.copy()
        
        # Drop the "Unnamed: 10" column if it exists
        if 'Unnamed: 10' in anime_features_copy.columns:
            anime_features_copy = anime_features_copy.drop('Unnamed: 10', axis=1)
        
        # Ensure we're using global_rating consistently
        if 'rating' in anime_features_copy.columns and 'global_rating' not in anime_features_copy.columns:
            anime_features_copy = anime_features_copy.rename(columns={'rating': 'global_rating'})
        
        # Define feature groups for better organization
        user_features = [
            'user_mean_rating', 
            'user_rating_count', 
            'user_rating_std', 
            'engagement_percentile', 
            'genre_diversity'
        ]
        
        # Anime content features that are safe to use
        anime_features = [
            'episodes', 
            'members', 
            'popularity_percentile',
            'global_rating'
        ]
        
        # Identify genre columns - anything that's not one of the basic columns and not a type column
        genre_columns = [col for col in anime_features_copy.columns 
                        if col not in ['anime_id', 'name', 'episodes', 'global_rating', 'members',
                                    'rating_count', 'user_avg_rating', 'user_median_rating', 
                                    'user_rating_std', 'popularity_percentile'] 
                        and not col.startswith('type_')]
        
        # Identify type columns
        type_columns = [col for col in anime_features_copy.columns if col.startswith('type_')]
        
        # Merge ratings with anime features
        train_features = pd.merge(
            train_df, 
            anime_features_copy,
            on='anime_id', 
            how='left'
        )
        
        test_features = pd.merge(
            test_df, 
            anime_features_copy,
            on='anime_id', 
            how='left'
        )
        
        # Remove normalized_rating if it exists (derived from target)
        if 'normalized_rating' in train_features.columns:
            print("Removing normalized_rating column to avoid data leakage")
            train_features = train_features.drop('normalized_rating', axis=1)
            test_features = test_features.drop('normalized_rating', axis=1)
        
        # Process top_genres column
        if 'top_genres' in train_features.columns:
            print("Processing top_genres column with MultiLabelBinarizer...")
            
            # Convert string representations of lists to actual lists
            train_features['top_genres'] = train_features['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
            test_features['top_genres'] = test_features['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
            
            # Create a MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            
            # Fit and transform training data
            genre_matrix = mlb.fit_transform(train_features['top_genres'])
            
            # Transform test data
            genre_test_matrix = mlb.transform(test_features['top_genres'])
            
            # Convert to DataFrames with genre names as columns
            genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
            genre_test_df = pd.DataFrame(genre_test_matrix, columns=mlb.classes_)
            
            # Add prefix to genre columns to avoid confusion
            genre_df = genre_df.add_prefix('user_genre_')
            genre_test_df = genre_test_df.add_prefix('user_genre_')
            
            # Drop the original top_genres column
            train_features = train_features.drop('top_genres', axis=1)
            test_features = test_features.drop('top_genres', axis=1)
            
            # Concatenate the one-hot encoded genres with the original features
            train_features = pd.concat([train_features.reset_index(drop=True), genre_df], axis=1)
            test_features = pd.concat([test_features.reset_index(drop=True), genre_test_df], axis=1)
        
        # Handle missing values
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)
        
        # Collect all features to use
        all_features = []
        
        # Add user features
        for feature in user_features:
            if feature in train_features.columns:
                all_features.append(feature)
        
        # Add anime features
        for feature in anime_features:
            if feature in train_features.columns:
                all_features.append(feature)
        
        # Add genre columns
        for feature in genre_columns:
            if feature in train_features.columns:
                all_features.append(feature)
        
        # Add type columns
        for feature in type_columns:
            if feature in train_features.columns:
                all_features.append(feature)
        
        # Add user genre preferences (from one-hot encoded top_genres)
        user_genre_columns = [col for col in train_features.columns if col.startswith('user_genre_')]
        all_features.extend(user_genre_columns)
        
        print("***************************************************************************")
        print(f"Using {len(all_features)} features for training")
        logger.info(f"Using {len(all_features)} features for training")
        logger.info(all_features)
        print("***************************************************************************")
        self.feature_columns = all_features
        
        # Create feature matrices and target vectors
        X_train = train_features[all_features]
        y_train = train_features['rating']  # This is the user's rating we want to predict
        
        X_test = test_features[all_features]
        y_test = test_features['rating']
        
        print(f"Features prepared. X_train shape: {X_train.shape}")
        logger.info(f"Features prepared. Using {len(all_features)} features.")
        
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        """Train the AdaBoost model."""
        print("Training AdaBoost model...")
        
        # Use a smaller sample if the dataset is very large
        sample_size = min(100000, len(X_train))
        if len(X_train) > sample_size:
            print(f"Using a sample of {sample_size} examples for grid search")
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]
        else:
            X_sample = X_train
            y_sample = y_train
            
        start_time = time.time()
        
        # Parameter grid for AdaBoost
        # - Base estimator is DecisionTreeRegressor with limited depth
        # - n_estimators is the number of weak learners to train
        # - learning_rate controls the contribution of each weak learner
        try:
            # Try creating a simple model to test which parameter naming is correct
            test_model = AdaBoostRegressor(estimator=DecisionTreeRegressor())
            # If we reach here, we're using the newer API
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0],
                'estimator__max_depth': [3, 5]
            }
            print("Using newer scikit-learn API (estimator parameter)")
        except TypeError:
            # Fall back to the older API
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0],
                'base_estimator__max_depth': [3, 5]
            }
            print("Using older scikit-learn API (base_estimator parameter)")
        
        # Create base estimator with limited depth to avoid overfitting
        base_estimator = DecisionTreeRegressor(random_state=42)
        
        # Initialize AdaBoost model
        # In newer scikit-learn versions, we use 'estimator' instead of 'base_estimator'
        try:
            # First try the newer API (scikit-learn >= 1.0)
            ada_base = AdaBoostRegressor(
                estimator=base_estimator,
                random_state=42
            )
        except TypeError:
            # Fall back to older API (scikit-learn < 1.0)
            ada_base = AdaBoostRegressor(
                base_estimator=base_estimator,
                random_state=42
            )
        
        # Grid search to find optimal parameters
        grid = GridSearchCV(
            estimator=ada_base,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid.fit(X_sample, y_sample)
        
        # Get best parameters
        best_params = grid.best_params_
        print(f"Best AdaBoost parameters: {best_params}")
        
        # Check if we're using 'base_estimator__max_depth' or 'estimator__max_depth'
        depth_param_name = None
        for param in best_params:
            if param.endswith('max_depth'):
                depth_param_name = param
                break
        
        if depth_param_name:
            # Extract the max_depth parameter
            base_max_depth = best_params.pop(depth_param_name)
            # Configure base estimator with optimal max_depth
            optimal_base_estimator = DecisionTreeRegressor(
                max_depth=base_max_depth,
                random_state=42
            )
            
            # Train on full dataset with best parameters
            try:
                # Try newer scikit-learn API first
                self.ada_model = AdaBoostRegressor(
                    estimator=optimal_base_estimator,
                    n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate'],
                    random_state=42
                )
            except TypeError:
                # Fall back to older API
                self.ada_model = AdaBoostRegressor(
                    base_estimator=optimal_base_estimator,
                    n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate'],
                    random_state=42
                )
        else:
            # If max_depth parameter wasn't found, use default estimator
            try:
                # Try newer scikit-learn API first
                self.ada_model = AdaBoostRegressor(
                    n_estimators=best_params.get('n_estimators', 100),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    random_state=42
                )
            except TypeError:
                # Fall back to older scikit-learn API
                self.ada_model = AdaBoostRegressor(
                    n_estimators=best_params.get('n_estimators', 100),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    random_state=42
                )
        
        self.ada_model.fit(X_train, y_train)
        
        print(f"AdaBoost model trained in {time.time() - start_time:.2f} seconds")
        
        # Save model
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/adaboost_model_regressor.pkl', 'wb') as f:
                pickle.dump(self.ada_model, f)
            print("AdaBoost model saved successfully")
        except Exception as e:
            print(f"Warning: Could not save model: {str(e)}")
            
        return self.ada_model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the AdaBoost model on test data."""
        if self.ada_model is None:
            print("Error: Model has not been trained. Call train_model() first.")
            return None
            
        print("\n===== ADABOOST MODEL EVALUATION =====")
        
        # Get predictions
        predictions = self.ada_model.predict(X_test)
        
        # Round and clip predictions for discrete evaluation
        predictions_rounded = np.round(np.clip(predictions, 1, 10))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        accuracy = np.mean(predictions_rounded == np.round(y_test))
        within_one = np.mean(abs(predictions_rounded - np.round(y_test)) <= 1)
        
        print(f"AdaBoost Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Accuracy (rounded): {accuracy:.4f}")
        print(f"Within 1 point: {within_one:.4f}")
        
        # Feature importance
        if hasattr(self.ada_model, 'feature_importances_'):
            importance_scores = self.ada_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 important features:")
            for i, (feature, importance) in enumerate(
                zip(importance_df['Feature'].head(10), 
                    importance_df['Importance'].head(10))):
                print(f"{i+1}. {feature}: {importance:.4f}")
        else:
            print("\nFeature importance not available for this model.")
        
        # Error analysis
        errors = y_test - predictions
        
        print("\nError Statistics:")
        print(f"Mean Error: {errors.mean():.4f}")
        print(f"Error Std Dev: {errors.std():.4f}")
        print(f"Min Error: {errors.min():.4f}")
        print(f"Max Error: {errors.max():.4f}")
        
        # Analyze errors by true rating
        rating_groups = pd.DataFrame({
            'true_rating': y_test,
            'error': np.abs(errors)
        }).groupby('true_rating')
        
        error_by_rating = rating_groups.mean()
        
        print("\nMean Absolute Error by True Rating:")
        print(error_by_rating)
        
        # Return evaluation results
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'within_one': within_one,
            'errors': errors,
            'error_by_rating': error_by_rating
        }
        
        # Plot error distribution
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=20, alpha=0.7, color='blue')
            plt.title('AdaBoost Error Distribution')
            plt.xlabel('Error (Predicted - Actual)')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.grid(True, alpha=0.3)
            plt.savefig('adaboost_error_distribution.png')
            print("\nError distribution plot saved to 'adaboost_error_distribution.png'")
        except Exception as e:
            print(f"Warning: Could not create error distribution plot: {str(e)}")
        
        return results

    def generate_recommendations(self, user_id, top_n=10):
        """Generate anime recommendations for a specific user."""
        if self.ada_model is None:
            print("Error: Model has not been trained.")
            return None
        
        try:
            # Find user in the training data
            user_data = self.rating_data[self.rating_data['user_id'] == user_id]
            
            if len(user_data) == 0:
                print(f"User {user_id} not found in training data")
                return None
                
            # Get anime the user has already rated
            rated_anime = set(user_data['anime_id'].unique())
            
            # Get all available anime
            all_anime = set(self.anime_features['anime_id'].unique())
            
            # Get unrated anime
            unrated_anime = list(all_anime - rated_anime)
            
            # If there are too many candidates, sample for efficiency
            if len(unrated_anime) > 1000:
                print(f"Sampling from {len(unrated_anime)} unrated anime for efficiency...")
                np.random.seed(42)
                unrated_anime = np.random.choice(unrated_anime, 1000, replace=False)
            
            print(f"Generating predictions for {len(unrated_anime)} unrated anime...")
            
            # Get user features
            user_features = {}
            for feature in user_data.columns:
                if feature.startswith('user_') or feature == 'engagement_percentile' or feature == 'genre_diversity':
                    user_features[feature] = user_data[feature].iloc[0]
            
            # Create candidate rows for prediction
            candidate_rows = []
            for anime_id in unrated_anime:
                anime_info = self.anime_features[self.anime_features['anime_id'] == anime_id]
                
                if anime_info.empty:
                    continue
                
                # Start with user features
                row = {'user_id': user_id, 'anime_id': anime_id}
                row.update(user_features)
                
                # Add anime features
                for col in anime_info.columns:
                    if col not in ['anime_id', 'name']:
                        if col == 'rating':
                            row['global_rating'] = anime_info[col].iloc[0]
                        else:
                            row[col] = anime_info[col].iloc[0]
                
                candidate_rows.append(row)
            
            # Convert to dataframe
            candidates_df = pd.DataFrame(candidate_rows)
            
            # Handle top_genres if needed
            user_genre_columns = [col for col in self.feature_columns if col.startswith('user_genre_')]
            if user_genre_columns:
                for col in user_genre_columns:
                    if col not in candidates_df.columns:
                        # Get the genre preference from user data if available
                        if col in user_data.columns:
                            candidates_df[col] = user_data[col].iloc[0]
                        else:
                            candidates_df[col] = 0
            
            # Ensure all model features are present
            for col in self.feature_columns:
                if col not in candidates_df.columns:
                    candidates_df[col] = 0
            
            # Keep only the features needed by the model
            X_candidates = candidates_df[self.feature_columns].fillna(0)
            
            # Generate predictions
            predictions = self.ada_model.predict(X_candidates)
            
            # Round and clip predictions
            predictions_rounded = np.round(np.clip(predictions, 1, 10)).astype(int)
            
            # Add predictions to candidates
            candidates_df['predicted_rating'] = predictions_rounded
            
            # Get anime names
            result_df = candidates_df[['anime_id', 'predicted_rating']].copy()
            result_df['name'] = result_df['anime_id'].apply(
                lambda x: self.anime_features.loc[self.anime_features['anime_id'] == x, 'name'].iloc[0]
                if not self.anime_features.loc[self.anime_features['anime_id'] == x, 'name'].empty
                else f"Unknown Anime {x}"
            )
            
            # Sort by predicted rating (descending)
            recommendations = result_df.sort_values('predicted_rating', ascending=False).head(top_n)
            
            print(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations[['anime_id', 'name', 'predicted_rating']]
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to train and evaluate the AdaBoost recommender."""
    print("Starting Anime Recommendation System using AdaBoost...")
    
    # Create an instance of the recommender
    recommender = AnimeAdaBoostRecommender()
    
    try:
        # Load data
        print("\n==== LOADING DATA ====")
        train_data, test_data = recommender.load_data()
        
        # Prepare features
        print("\n==== PREPARING FEATURES ====")
        X_train, y_train, X_test, y_test = recommender.prepare_features(train_data, test_data)
        
        # Train the AdaBoost model
        print("\n==== TRAINING ADABOOST MODEL ====")
        ada_model = recommender.train_model(X_train, y_train)
        
        # Evaluate model
        print("\n==== EVALUATING MODEL ====")
        results = recommender.evaluate_model(X_test, y_test)
        
        # Generate sample recommendations
        if len(recommender.rating_data['user_id'].unique()) > 0:
            sample_user = recommender.rating_data['user_id'].unique()[0]
            print(f"\n==== SAMPLE RECOMMENDATIONS ====")
            print(f"Generating recommendations for sample user {sample_user}...")
            recommendations = recommender.generate_recommendations(sample_user, top_n=5)
            
            if recommendations is not None:
                print("\nTop 5 Recommendations:")
                print(recommendations.to_string(index=False))
        
        print("\nAdaBoost recommendation system completed successfully!")
        
    except Exception as e:
        print(f"Error in AdaBoost recommender: {str(e)}")
        import traceback
        traceback.print_exc()
        print("AdaBoost recommendation system failed to complete.")
    
if __name__ == "__main__":
    main()