import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import logging
import random
import joblib
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="anime_knn_classifier.log"
)
logger = logging.getLogger(__name__)

class AnimeKNNClassifier:
    """KNN-based anime recommendation system using classification approach."""
    
    def __init__(self, n_neighbors=10, n_components=20, use_pca=True):
        """
        Initialize the KNN classification system.
        
        Parameters:
        -----------
        n_neighbors : int, default=10
            Number of neighbors to use
        n_components : int, default=20
            Number of PCA components to reduce dimensionality
        use_pca : bool, default=True
            Whether to use PCA for dimensionality reduction
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.use_pca = use_pca
        self.model = None
        self.pca = None
        self.scaler = None
        self.feature_columns = None
        self.optimization_size = 100000  # Use 100k samples for k optimization
        self.anime_dict = {}  # For quick lookups
    
    def load_data(self, data_path='data/processed/'):
        """Load the necessary data files for the recommendation system."""
        print("Loading data files...")
        
        try:
            # Load the preprocessed data files
            self.anime_features = pd.read_csv(os.path.join("..", data_path, 'anime_features_normalized_new.csv'))
            
            # Load train and test sets
            train_data = pd.read_csv(os.path.join("..", data_path, 'rating_train_v2_new.csv'))
            test_data = pd.read_csv(os.path.join("..", data_path, 'rating_test_v2_new.csv'))
            
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
        
        # Convert continuous ratings to discrete classes for classification
        y_train = train_features['rating'].round().astype(int)
        
        X_test = test_features[all_features]
        y_test = test_features['rating'].round().astype(int)
        
        # Store the unique rating classes
        self.classes_ = sorted(y_train.unique())
        
        # Check class distribution
        print("\nClass distribution in training set:")
        train_class_dist = y_train.value_counts().sort_index()
        for rating, count in train_class_dist.items():
            percentage = count / len(y_train) * 100
            print(f"Rating {rating}: {count} samples ({percentage:.2f}%)")
        
        # Scale the features for KNN
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA if requested
        if self.use_pca:
            print(f"Applying PCA to reduce dimensions to {self.n_components} components")
            self.pca = PCA(n_components=self.n_components)
            X_train_processed = self.pca.fit_transform(X_train_scaled)
            X_test_processed = self.pca.transform(X_test_scaled)
            
            # Log the explained variance ratio
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA explained variance: {explained_variance:.4f}")
        else:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
        
        print(f"Features prepared. X_train shape: {X_train_processed.shape}")
        logger.info(f"Features prepared. Using {len(all_features)} features.")
        
        return X_train_processed, y_train, X_test_processed, y_test
    
    def create_optimization_dataset(self, X_train, y_train):
        """Create a subset of data for k optimization."""
        print(f"Creating optimization dataset with {self.optimization_size} samples...")
        
        # If we have more than optimization_size training samples, take a random subset
        if len(X_train) > self.optimization_size:
            # Use train_test_split to create a stratified sample
            X_opt, _, y_opt, _ = train_test_split(
                X_train, y_train, 
                train_size=self.optimization_size,
                random_state=42,
                stratify=y_train  # Stratify by rating class
            )
        else:
            X_opt = X_train
            y_opt = y_train
        
        print(f"Optimization dataset created with {len(X_opt)} samples")
        
        # Split optimization set into train and validation
        X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
            X_opt, y_opt, test_size=0.2, random_state=42, stratify=y_opt
        )
        
        return X_opt_train, y_opt_train, X_opt_val, y_opt_val
    
    def train_model(self, X_train, y_train):
        """Train the KNN classifier model."""
        print(f"Training KNN classifier with {self.n_neighbors} neighbors on {len(X_train)} samples...")
        start_time = time.time()
        
        # Initialize the model
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2,
            n_jobs=-1  # Use all cores
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.2f} seconds")
        return train_time
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance using classification metrics."""
        if self.model is None:
            print("Error: Model not trained")
            return None
        
        print(f"Evaluating model performance on {len(X_test)} samples...")
        start_time = time.time()
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Define the class labels (possible ratings)
        labels = sorted(y_test.unique())
        
        # Calculate standard classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, labels=labels, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, labels=labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=labels, average='weighted', zero_division=0)
        
        # Calculate regression-like metrics - RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate "within 1 point" accuracy (predictions within 1 rating point)
        within_one = np.mean(abs(y_pred - y_test) <= 1)
        
        # Build confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        eval_time = time.time() - start_time
        
        # Print results
        print("\n===== EVALUATION RESULTS =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Within 1 point: {within_one:.4f}")
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'rmse': rmse,
            'mae': mae,
            'within_one': within_one,
            'confusion_matrix': cm,
            'eval_time': eval_time
        }
    
    def optimize_k(self, X_train, y_train, X_val, y_val, k_values=[5, 10, 15, 20, 30, 50]):
        """Find the optimal k value using a validation set."""
        print("Optimizing k value...")
        
        # Grid search
        param_grid = {'n_neighbors': k_values}
        
        # Create KNN model
        knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
        
        # Use grid search with 3-fold CV
        grid_search = GridSearchCV(
            knn, 
            param_grid, 
            cv=3, 
            scoring='accuracy',  # Use accuracy for classification
            n_jobs=-1
        )
        
        # Fit to optimization set
        grid_search.fit(X_train, y_train)
        
        # Best k value and results
        best_k = grid_search.best_params_['n_neighbors']
        results = grid_search.cv_results_
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'k': k_values,
            'mean_accuracy': results['mean_test_score'],
            'std_accuracy': results['std_test_score'],
            'mean_fit_time': results['mean_fit_time'],
            'mean_score_time': results['mean_score_time']
        })
        
        # Print results
        print("\nK optimization results:")
        print(results_df)
        print(f"\nBest k value: {best_k}")
        
        # Set the optimal k value
        self.n_neighbors = best_k
        
        return best_k, results_df
    
    def analyze_class_performance(self, X_test, y_test):
        """Analyze model performance by class (rating value)."""
        print("\nAnalyzing performance by rating class...")
        
        # Model predictions
        y_pred = self.model.predict(X_test)
        
        # Create dataframe with true and predicted values
        results_df = pd.DataFrame({
            'true_rating': y_test,
            'pred_rating': y_pred
        })
        
        # Calculate per-class accuracy
        class_accuracy = {}
        
        unique_ratings = sorted(y_test.unique())
        
        print("\nPer-Class Accuracy:")
        print(f"{'Rating':<10} {'Accuracy':<15} {'Sample Count':<15}")
        print("-" * 40)
        
        for rating in unique_ratings:
            class_mask = (y_test == rating)
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            sample_count = np.sum(class_mask)
            
            class_accuracy[rating] = class_acc
            
            print(f"{rating:<10} {class_acc:<15.4f} {sample_count:<15}")
        
        # Calculate "within 1" accuracy by class
        within_one = {}
        
        print("\nPer-Class 'Within 1' Accuracy:")
        print(f"{'Rating':<10} {'Within 1':<15} {'Sample Count':<15}")
        print("-" * 40)
        
        for rating in unique_ratings:
            class_mask = (y_test == rating)
            within_one_acc = np.mean(abs(y_pred[class_mask] - rating) <= 1)
            sample_count = np.sum(class_mask)
            
            within_one[rating] = within_one_acc
            
            print(f"{rating:<10} {within_one_acc:<15.4f} {sample_count:<15}")
        
        return {
            'class_accuracy': class_accuracy,
            'within_one': within_one
        }
    
    def generate_recommendations(self, user_id, top_n=10):
        """Generate anime recommendations for a specific user."""
        if self.model is None:
            print("Error: Model not trained")
            return None
        
        try:
            # Create sample user data from training set
            user_data = None
            for train_df in [self.train_data, self.test_data]:
                if user_id in train_df['user_id'].values:
                    user_data = train_df[train_df['user_id'] == user_id]
                    break
            
            if user_data is None or len(user_data) == 0:
                print(f"User {user_id} not found in data")
                return None
            
            # Find anime the user hasn't rated
            rated_anime = set(user_data['anime_id'].unique())
            all_anime = set(self.anime_features['anime_id'].unique())
            unrated_anime = list(all_anime - rated_anime)
            
            # Take a sample of unrated anime for faster processing
            if len(unrated_anime) > 1000:
                unrated_anime = random.sample(unrated_anime, 1000)
            
            # Create a test dataset for prediction
            test_rows = []
            
            # Use a sample row from user data as template
            sample_user_row = user_data.iloc[0].copy()
            
            # Create test rows for each unrated anime
            for anime_id in unrated_anime:
                # Create a copy of the template row
                new_row = sample_user_row.copy()
                
                # Update anime_id
                new_row['anime_id'] = anime_id
                
                # Add to test rows
                test_rows.append(new_row)
            
            # Convert to dataframe
            test_df = pd.DataFrame(test_rows)
            
            # Prepare features using the same preprocessing pipeline
            test_features = pd.merge(
                test_df,
                self.anime_features,
                on='anime_id',
                how='left'
            )
            
            # Process top_genres if needed
            if 'top_genres' in test_features.columns:
                # Process top_genres same way as in prepare_features
                test_features = test_features.drop('top_genres', axis=1)
            
            # Extract features
            X_test = test_features[self.feature_columns].fillna(0)
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply PCA if used
            if self.use_pca and self.pca is not None:
                X_test_processed = self.pca.transform(X_test_scaled)
            else:
                X_test_processed = X_test_scaled
            
            # Predict ratings using classifier
            predictions = self.model.predict(X_test_processed)
            
            # For recommender systems, it's useful to have probabilities as well
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X_test_processed)
                
                # Create a confidence score as the probability of the predicted class
                confidences = []
                for i, pred in enumerate(predictions):
                    # Find the index of the predicted class
                    class_idx = list(self.model.classes_).index(pred)
                    confidences.append(probas[i, class_idx])
            else:
                # If probabilities aren't available, use dummy confidence
                confidences = [1.0] * len(predictions)
            
            # Create recommendations dataframe
            recommendations = pd.DataFrame({
                'anime_id': test_df['anime_id'],
                'predicted_rating': predictions,
                'confidence': confidences
            })
            
            # Add anime names
            anime_names = self.anime_features[['anime_id', 'name']].set_index('anime_id')
            recommendations = recommendations.merge(
                anime_names, 
                left_on='anime_id', 
                right_index=True,
                how='left'
            )
            
            # Sort by predicted rating first, then by confidence
            recommendations = recommendations.sort_values(
                ['predicted_rating', 'confidence'], ascending=[False, False]
            ).head(top_n)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            print(f"Error generating recommendations: {e}")
            return None
    
    def save_model(self, filepath='anime_knn_classifier', directory='models', format='joblib'):
        """Save the trained model and its components to disk."""
        if self.model is None:
            print("Error: No trained model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            
            # Add extension based on format
            if format.lower() == 'pkl':
                full_path = os.path.join(directory, f"{filepath}.pkl")
                with open(full_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'pca': self.pca,
                        'scaler': self.scaler,
                        'feature_columns': self.feature_columns,
                        'n_neighbors': self.n_neighbors,
                        'n_components': self.n_components,
                        'use_pca': self.use_pca,
                        'classes_': self.classes_
                    }, f)
            else:  # Default to joblib
                full_path = os.path.join(directory, f"{filepath}.joblib")
                joblib.dump({
                    'model': self.model,
                    'pca': self.pca,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'n_neighbors': self.n_neighbors,
                    'n_components': self.n_components,
                    'use_pca': self.use_pca,
                    'classes_': self.classes_
                }, full_path)
                
            print(f"Model successfully saved to {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            print(f"Error saving model: {e}")
            return False
            
    def load_model(self, filepath, directory='models'):
        """Load a trained model and its components from disk."""
        try:
            full_path = os.path.join(directory, filepath)
            
            # Determine format based on file extension
            if full_path.endswith('.pkl'):
                with open(full_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:  # Default to joblib
                model_data = joblib.load(full_path)
            
            # Restore model components
            self.model = model_data['model']
            self.pca = model_data['pca']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.n_neighbors = model_data['n_neighbors']
            self.n_components = model_data['n_components']
            self.use_pca = model_data.get('use_pca', True)  # Default to True for backwards compatibility
            self.classes_ = model_data.get('classes_', [])
            
            print(f"Model successfully loaded from {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to run the KNN classification recommendation system."""
    print("Starting KNN classifier for anime recommendation...")
    
    # Create recommender
    recommender = AnimeKNNClassifier(
        n_neighbors=15,  # Initial value, will be optimized
        n_components=20,
        use_pca=True  # Use PCA for dimensionality reduction
    )
    
    try:
        # Load the data
        train_data, test_data = recommender.load_data()
        recommender.train_data = train_data  # Store for recommendation generation
        recommender.test_data = test_data
        
        # Prepare features with the DTRF-style preprocessing
        X_train, y_train, X_test, y_test = recommender.prepare_features(train_data, test_data)
        
        # Create optimization dataset
        X_opt_train, y_opt_train, X_opt_val, y_opt_val = recommender.create_optimization_dataset(X_train, y_train)
        
        # Find optimal k
        best_k, k_results = recommender.optimize_k(
            X_opt_train, y_opt_train, X_opt_val, y_opt_val,
            k_values=[5, 10, 15, 20, 30, 50]
        )
        
        # Train the final model with optimal k
        train_time = recommender.train_model(X_train, y_train)
        
        # Evaluate the model
        eval_results = recommender.evaluate(X_test, y_test)
        
        # Analyze class-specific performance
        class_analysis = recommender.analyze_class_performance(X_test, y_test)
        
        # Save the model
        recommender.save_model('anime_knn_classifier', directory='models')
        
        # Generate sample recommendations
        if len(train_data['user_id'].unique()) > 0:
            sample_user = train_data['user_id'].unique()[0]
            print(f"\nGenerating recommendations for sample user {sample_user}...")
            recommendations = recommender.generate_recommendations(sample_user, top_n=5)
            
            if recommendations is not None:
                print("\nTop 5 Recommendations:")
                print(recommendations[['name', 'predicted_rating', 'confidence']].to_string(index=False))
        
        print("\nKNN classification analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in KNN classifier: {e}")
        print(f"Error in KNN classifier: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 