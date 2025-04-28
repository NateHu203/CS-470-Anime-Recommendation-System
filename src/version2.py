import pandas as pd
import numpy as np
import os
import logging
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anime_recommender.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimeRecommender:
    """Anime recommendation system using decision trees/random forest."""
    
    def __init__(self, data_path: str = 'data/processed/'):
        """Initialize the recommendation system.
        
        Args:
            data_path: Path to the directory containing processed data files
        """
        self.data_path = data_path
        self.model = None
        self.anime_features = None
        self.rating_enhanced = None
        self.feature_columns = None
        self.all_features = None
        self.feature_importance = None
        self.anime_lookup = {}  # For quick access to anime data
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and testing data.
        
        Returns:
            Tuple of (train_rating, test_rating) DataFrames
        
        Raises:
            FileNotFoundError: If any required data file is not found
        """
        try:
            logger.info("Loading data...")
            start_time = time.time()
            
            # Load preprocessed data
            self.anime_features = pd.read_csv(os.path.join("..", self.data_path, 'anime_features_normalized.csv'))
            self.rating_enhanced = pd.read_csv(os.path.join("..", self.data_path, 'rating_enhanced.csv'))
            
            # Load training and testing data
            train_rating = pd.read_csv(os.path.join("..", self.data_path, 'rating_train_v2.csv'))
            test_rating = pd.read_csv(os.path.join("..", self.data_path, 'rating_test_v2.csv'))
            
            # Create a lookup dictionary for anime
            self.anime_lookup = self.anime_features.set_index('anime_id').to_dict('index')
            
            # Check if expected columns exist in rating_enhanced
            expected_columns = ['user_mean_rating', 'user_median_rating', 'user_rating_count', 
                              'user_rating_std', 'engagement_percentile', 'genre_diversity']
            
            missing_columns = [col for col in expected_columns if col not in self.rating_enhanced.columns]
            if missing_columns:
                logger.warning(f"Missing expected columns in rating_enhanced: {missing_columns}")
                
                # Add missing columns with calculated values if possible
                if 'user_median_rating' in missing_columns and 'rating' in self.rating_enhanced.columns:
                    logger.info("Calculating missing user_median_rating column...")
                    median_ratings = self.rating_enhanced.groupby('user_id')['rating'].median().reset_index()
                    median_ratings.columns = ['user_id', 'user_median_rating']
                    self.rating_enhanced = pd.merge(self.rating_enhanced, median_ratings, on='user_id', how='left')
                
                if 'user_rating_std' in missing_columns and 'rating' in self.rating_enhanced.columns:
                    logger.info("Calculating missing user_rating_std column...")
                    std_ratings = self.rating_enhanced.groupby('user_id')['rating'].std().reset_index()
                    std_ratings.columns = ['user_id', 'user_rating_std']
                    self.rating_enhanced = pd.merge(self.rating_enhanced, std_ratings, on='user_id', how='left')
                    self.rating_enhanced['user_rating_std'] = self.rating_enhanced['user_rating_std'].fillna(0)
            
            logger.info(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
            logger.info(f"Training set size: {len(train_rating)}, Test set size: {len(test_rating)}")
            logger.info(f"Available columns in rating_enhanced: {self.rating_enhanced.columns.tolist()}")
            return train_rating, test_rating
            
        except FileNotFoundError as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare features for model training.
        
        Args:
            train_df: Training data DataFrame
            test_df: Testing data DataFrame
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Preparing features...")
        start_time = time.time()
        
        # Verify that 'rating' column exists in the input data
        if 'rating' not in train_df.columns:
            logger.error("'rating' column not found in training data.")
            logger.info(f"Available columns in training data: {train_df.columns.tolist()}")
            raise KeyError("'rating' column not found in training data.")
            
        # Check for columns in anime_features
        logger.info(f"Available columns in anime_features: {self.anime_features.columns.tolist()}")
        
        # Merge anime features with ratings data to check available columns
        sample_df = pd.merge(train_df.head(1), self.anime_features, on='anime_id', how='left')
        available_columns = set(sample_df.columns)
        
        # Define feature columns based on available columns
        user_features = [col for col in [
            'user_mean_rating', 'user_rating_count', 'engagement_percentile', 'genre_diversity'
        ] if col in available_columns]
        
        anime_features = [col for col in [
            'episodes', 'members', 'rating_count', 'user_avg_rating',
            'popularity_percentile'
        ] if col in available_columns]
        
        # Define feature columns using only available columns
        self.feature_columns = user_features + anime_features
        
        # Merge anime features with ratings data using a more efficient approach
        train_with_features = pd.merge(train_df, self.anime_features, on='anime_id', how='left')
        test_with_features = pd.merge(test_df, self.anime_features, on='anime_id', how='left')
        
        # Ensure that rating column exists after merge
        if 'rating' not in train_with_features.columns:
            logger.error("'rating' column not found in merged training data.")
            logger.info(f"Available columns after merge: {train_with_features.columns.tolist()}")
            
            # If 'rating' column was in original dataframe but not in merged one,
            # it might have been overwritten by a column with the same name from anime_features
            if 'rating' in train_df.columns:
                logger.info("Restoring 'rating' column from original training data...")
                # Add a suffix to the rating column from anime_features if it exists
                if 'rating' in self.anime_features.columns:
                    train_with_features = pd.merge(
                        train_df, self.anime_features, on='anime_id', 
                        how='left', suffixes=('', '_anime')
                    )
                    test_with_features = pd.merge(
                        test_df, self.anime_features, on='anime_id', 
                        how='left', suffixes=('', '_anime')
                    )
            else:
                raise KeyError("'rating' column not found in merged training data.")
        
        # Find genre and type columns
        genre_columns = [col for col in self.anime_features.columns if col not in [
            'anime_id', 'name', 'episodes', 'rating', 'members', 
            'rating_count', 'user_avg_rating', 'popularity_percentile'
        ] and not col.startswith('type_') and col in train_with_features.columns]
        
        type_columns = [col for col in self.anime_features.columns if col.startswith('type_') 
                       and col in train_with_features.columns]
        
        # Combine all features and verify they exist in the dataset
        self.all_features = []
        for col in self.feature_columns + genre_columns + type_columns:
            if col in train_with_features.columns:
                self.all_features.append(col)
        
        logger.info(f"Using {len(self.all_features)} validated features from {len(self.feature_columns)} base features, "
                   f"{len(genre_columns)} genre features, and {len(type_columns)} type features")
        
        logger.info(f'Used {self.all_features}\n\n as traning feautres\n\n ')
        # Handle missing values
        train_with_features = train_with_features.fillna(0)
        test_with_features = test_with_features.fillna(0)
        
        # Extract features and labels
        X_train = train_with_features[self.all_features]
        y_train = train_with_features['rating']
        
        X_test = test_with_features[self.all_features]
        y_test = test_with_features['rating']
        
        logger.info(f"Feature preparation completed in {time.time() - start_time:.2f} seconds.")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
       
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'decision_tree') -> Any:
        """Train either a Decision Tree or Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Either 'decision_tree' or 'random_forest'
            
        Returns:
            Trained model
            
        Raises:
            ValueError: If invalid model_type is provided
        """
        logger.info(f"Training {model_type} model...")
        start_time = time.time()
        
        # Handle large datasets by using a sample for hyperparameter tuning
        sample_size = min(100000, len(X_train))
        use_sample = len(X_train) > sample_size
        
        if use_sample:
            logger.info(f"Using a sample of {sample_size} examples for hyperparameter tuning")
            # Create a stratified sample based on rating values
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        if model_type == 'decision_tree':
            # Parameters for grid search
            param_grid = {
                'max_depth': [10, 15],
                'min_samples_split': [10],
                'min_samples_leaf': [4]
            }
            base_model = DecisionTreeRegressor(random_state=42)
            
        elif model_type == 'random_forest':
            # Reduced parameter grid for faster training with large datasets
            param_grid = {
                'n_estimators': [50],
                'max_depth': [10, 15],
                'min_samples_split': [10],
                'min_samples_leaf': [4]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'decision_tree' or 'random_forest'.")
        
        # Grid search with cross-validation
        logger.info("Performing grid search for optimal parameters...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        
        try:
            # Fit the model on the sample data
            grid_search.fit(X_sample, y_sample)
            
            # Get best parameters
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            # Train final model on full dataset with best parameters
            if use_sample:
                logger.info("Training final model on full dataset with best parameters...")
                if model_type == 'decision_tree':
                    self.model = DecisionTreeRegressor(random_state=42, **best_params)
                else:
                    self.model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
                
                self.model.fit(X_train, y_train)
            else:
                self.model = grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            logger.warning("Falling back to default parameters...")
            
            # Fall back to default model with minimal parameters
            if model_type == 'decision_tree':
                self.model = DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train, y_train)
        
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds.")
        
        # Save the model
        os.makedirs('models_test', exist_ok=True)
        model_filename = f"models_test/{model_type}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_filename}")
        
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        logger.info("Evaluating model...")
        start_time = time.time()
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Test set results: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Analyze feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance rankings
            
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        logger.info("Analyzing feature importance...")
        
        # Get feature importance (handling length mismatch if it occurs)
        importances = self.model.feature_importances_
        
        # Make sure we have the correct set of features
        features_used = self.all_features
        if len(importances) != len(features_used):
            logger.warning(f"Feature length mismatch: {len(importances)} importances vs {len(features_used)} features")
            # Use the feature names from the model if available
            if hasattr(self.model, 'feature_names_in_'):
                features_used = self.model.feature_names_in_
            else:
                # Truncate to match lengths if needed
                features_used = features_used[:len(importances)]
        
        # Create DataFrame with importance values
        self.feature_importance = pd.DataFrame({
            'Feature': features_used,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Save feature importance
        self.feature_importance.to_csv('models/feature_importance.csv', index=False)
        
        # Log top features
        logger.info("Top 10 important features:")
        for i, (feature, importance) in enumerate(
            zip(self.feature_importance['Feature'].head(10), 
                self.feature_importance['Importance'].head(10))):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        return self.feature_importance
    
    def _batch_process_anime(self, batch_anime_ids: List[int], user_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Process a batch of anime for recommendations.
        
        Args:
            batch_anime_ids: List of anime IDs to process
            user_features: Dictionary of user features
            
        Returns:
            List of recommendation dictionaries with predictions
        """
        recommendations = []
        
        # Get feature names required by the model
        model_features = []
        if hasattr(self.model, 'feature_names_in_'):
            model_features = list(self.model.feature_names_in_)
            logger.info(f"Model expects {len(model_features)} features: {model_features[:5]}...")
        
        # Process anime in smaller chunks to avoid large array creation
        chunk_size = 1000  # Smaller chunks within a batch
        for i in range(0, len(batch_anime_ids), chunk_size):
            chunk_ids = batch_anime_ids[i:i+chunk_size]
            
            # Create feature vectors for all anime in this chunk
            chunk_features = []
            chunk_names = []
            chunk_valid_ids = []
            
            for anime_id in chunk_ids:
                if anime_id not in self.anime_lookup:
                    continue
                    
                anime = self.anime_lookup[anime_id]
                
                # Create feature vector
                features = user_features.copy()
                
                # Add anime features
                for feature in self.all_features:
                    if feature not in features and feature in anime:
                        features[feature] = anime[feature]
                    elif feature not in features:
                        features[feature] = 0
                
                chunk_features.append(features)
                chunk_names.append(anime.get('name', f"Anime {anime_id}"))
                chunk_valid_ids.append(anime_id)
            
            try:
                if not chunk_features:
                    continue
                    
                # Convert to DataFrame for prediction
                X = pd.DataFrame(chunk_features)
                
                # If we have model_features, ensure we use exactly those
                if model_features:
                    # First, ensure all required features exist (add missing ones with 0)
                    for feature in model_features:
                        if feature not in X.columns:
                            X[feature] = 0
                    
                    # Then, select only the features required by the model in the correct order
                    X = X[model_features]
                
                # Make predictions for the chunk
                predictions = self.model.predict(X)
                
                # Create recommendation objects
                for anime_id, name, pred in zip(chunk_valid_ids, chunk_names, predictions):
                    recommendations.append({
                        'anime_id': anime_id,
                        'name': name,
                        'predicted_rating': float(pred)  # Ensure it's a Python float not numpy type
                    })
                    
            except Exception as e:
                logger.error(f"Error predicting chunk of anime: {e}")
                # Log more detailed information to help debug
                if model_features and len(model_features) > 0:
                    missing_features = [f for f in model_features if f not in X.columns]
                    extra_features = [f for f in X.columns if f not in model_features]
                    logger.error(f"Missing features: {missing_features}")
                    logger.error(f"Extra features: {extra_features}")
                # Continue with other chunks
        
        return recommendations
    
    def recommend_for_user(self, user_id: Optional[int] = None, 
                          preferences: Optional[Dict[str, Any]] = None,
                          watched_anime_ids: Optional[List[int]] = None,
                          top_n: int = 10,
                          batch_size: int = 5000,
                          use_parallel: bool = True,
                          max_workers: int = 4) -> List[Dict[str, Any]]:
        """Generate recommendations for a user.
        
        This unified function handles both existing and new users.
        
        Args:
            user_id: ID of existing user (None for new users)
            preferences: Dictionary of preferences for new users
            watched_anime_ids: List of anime IDs already watched
            top_n: Number of recommendations to return
            batch_size: Size of batches for processing
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes to use
            
        Returns:
            List of top recommendations
            
        Raises:
            ValueError: If both user_id and preferences are None, or if model is not trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        if user_id is None and preferences is None:
            raise ValueError("Either user_id or preferences must be provided.")
            
        start_time = time.time()
        
        if user_id is not None:
            # Existing user
            logger.info(f"Generating recommendations for existing user {user_id}...")
            
            # Get user features
            try:
                user_data = self.rating_enhanced[self.rating_enhanced['user_id'] == user_id].iloc[0].to_dict()
                
                # Get watched anime if not provided
                if watched_anime_ids is None:
                    watched_anime_ids = self.rating_enhanced[
                        self.rating_enhanced['user_id'] == user_id
                    ]['anime_id'].tolist()
            except (IndexError, KeyError) as e:
                logger.error(f"User {user_id} not found in data: {e}")
                return []
                
            # Extract user features (only use features that are available)
            user_feature_names = ['user_mean_rating', 'user_rating_count',
                                 'engagement_percentile', 'genre_diversity']
            user_features = {}
            for feature in user_feature_names:
                if feature in user_data:
                    user_features[feature] = user_data.get(feature, 0)
                else:
                    user_features[feature] = 0
            
        else:
            # New user with preferences
            logger.info("Generating recommendations for new user based on preferences...")
            
            # Calculate average user stats for baseline (only for columns that exist)
            available_columns = [col for col in ['user_mean_rating', 'user_rating_count'] 
                               if col in self.rating_enhanced.columns]
            
            avg_user_stats = {}
            if available_columns:
                avg_user_stats = self.rating_enhanced[available_columns].mean().to_dict()
            
            # Use median for some metrics if available
            if 'user_rating_count' in self.rating_enhanced.columns:
                avg_user_stats['user_rating_count'] = self.rating_enhanced['user_rating_count'].median()
            
            # Set default values for common features
            avg_user_stats['engagement_percentile'] = 50  # Middle value
            
            # Set genre diversity based on preferences
            avg_user_stats['genre_diversity'] = len(preferences.get('genres', []))
            
            user_features = avg_user_stats
            watched_anime_ids = []
        
        # Get all unwatched anime IDs
        all_anime_ids = list(self.anime_lookup.keys())
        unwatched_anime_ids = [aid for aid in all_anime_ids if aid not in watched_anime_ids]
        
        # Split into batches (use larger batches for efficiency with high RAM)
        batches = [unwatched_anime_ids[i:i + batch_size] 
                  for i in range(0, len(unwatched_anime_ids), batch_size)]
        
        logger.info(f"Processing {len(unwatched_anime_ids)} unwatched anime in {len(batches)} batches")
        
        # Store results as priority queue to keep top recommendations
        all_recommendations = []
        
        try:
            if use_parallel and len(batches) > 1:
                # Configure process pool to use fewer workers but larger batches
                # This reduces memory duplication while still leveraging parallelism
                n_workers = min(max_workers, len(batches))
                logger.info(f"Using parallel processing with {n_workers} workers")
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Process batches in parallel
                    futures = []
                    for batch in batches:
                        future = executor.submit(self._batch_process_anime, batch, user_features)
                        futures.append(future)
                    
                    # Collect results as they complete and maintain top recommendations
                    for i, future in enumerate(futures):
                        try:
                            batch_recommendations = future.result()
                            
                            # Sort and keep only top recommendations from each batch
                            if batch_recommendations:
                                batch_recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
                                best_from_batch = batch_recommendations[:top_n*2]
                                all_recommendations.extend(best_from_batch)
                                
                                # Periodically trim recommendations to save memory
                                if len(all_recommendations) > top_n * 10:
                                    all_recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
                                    all_recommendations = all_recommendations[:top_n * 5]
                            
                            logger.info(f"Processed batch {i+1}/{len(batches)}")
                        except Exception as e:
                            logger.error(f"Error processing batch {i+1}: {e}")
            else:
                # Process sequentially
                logger.info("Using sequential processing")
                for i, batch in enumerate(batches):
                    batch_recommendations = self._batch_process_anime(batch, user_features)
                    
                    # Keep top recommendations
                    if batch_recommendations:
                        batch_recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
                        all_recommendations.extend(batch_recommendations[:top_n*2])
                        
                        # Periodically trim the list to save memory
                        if len(all_recommendations) > top_n * 10:
                            all_recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
                            all_recommendations = all_recommendations[:top_n * 5]
                    
                    logger.info(f"Processed batch {i+1}/{len(batches)}")
                    
        except Exception as e:
            logger.error(f"Error during recommendation generation: {e}")
            # Continue with any recommendations already collected
        
        # Apply preference boosting for new users
        if preferences:
            # Boost scores for preferred genres and types
            for rec in all_recommendations:
                anime_id = rec['anime_id']
                anime = self.anime_lookup.get(anime_id, {})
                
                # Boost for preferred genres
                for genre in preferences.get('genres', []):
                    if genre in anime and anime[genre] == 1:
                        rec['predicted_rating'] += 0.5  # Boost score
                        
                # Boost for preferred type
                preferred_type = preferences.get('type', '')
                type_col = f"type_{preferred_type}"
                if type_col in anime and anime[type_col] == 1:
                    rec['predicted_rating'] += 0.5  # Boost score
        
        # Final sort of recommendations by predicted rating
        all_recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        logger.info(f"Generated {len(all_recommendations)} recommendations in {time.time() - start_time:.2f} seconds")
        
        # Return top N recommendations
        return all_recommendations[:top_n]
    
    def load_saved_model(self, model_path: str) -> None:
        """Load a previously saved model.
        
        Args:
            model_path: Path to the saved model file
            
        Raises:
            FileNotFoundError: If the model file is not found
        """
        try:
            logger.info(f"Loading model from {model_path}...")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def main():
    """Main function to run the anime recommendation system."""
    try:
        # Initialize the recommender
        recommender = AnimeRecommender()
        
        # Load data
        train_rating, test_rating = recommender.load_data()
        
        # Memory optimization: reduce DataFrame memory usage
        for df in [train_rating, test_rating, recommender.anime_features, recommender.rating_enhanced]:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    df[col] = df[col].astype('int32')
        
        # Prepare features
        X_train, y_train, X_test, y_test = recommender.prepare_features(train_rating, test_rating)
        
        # Train model (use decision_tree for faster training and less memory usage)
        model = recommender.train_model(X_train, y_train, model_type='random_forest')
        
        # Evaluate model
        evaluation = recommender.evaluate_model(X_test, y_test)
        
        # Get feature importance
        feature_importance = recommender.get_feature_importance()
        
        # Example: Generate recommendations for existing user
        example_user_id = train_rating['user_id'].iloc[0]
        
        # Use larger batch size to process more anime at once
        existing_user_recommendations = recommender.recommend_for_user(
            user_id=example_user_id, 
            top_n=5,
            batch_size=10000,  # Larger batch size for systems with more RAM
            use_parallel=True,
            max_workers=2      # Use fewer workers to reduce memory duplication
        )
        
        logger.info("\nExample recommendations for existing user:")
        for i, rec in enumerate(existing_user_recommendations, 1):
            logger.info(f"{i}. {rec['name']} (Predicted rating: {rec['predicted_rating']:.2f})")
        
        # Example: Generate recommendations for new user
        example_preferences = {
            'genres': ['Action', 'Adventure', 'Fantasy'],
            'type': 'TV'
        }
        
        new_user_recommendations = recommender.recommend_for_user(
            preferences=example_preferences, 
            top_n=5,
            batch_size=10000,  # Larger batch size
            use_parallel=True,
            max_workers=2      # Fewer workers
        )
        
        logger.info("\nExample recommendations for new user:")
        for i, rec in enumerate(new_user_recommendations, 1):
            logger.info(f"{i}. {rec['name']} (Predicted rating: {rec['predicted_rating']:.2f})")
        
        logger.info("\nRecommendation system completed!")
        
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()