import pandas as pd
import numpy as np
import os
import logging
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
    filename="anime_recommender_version5.log"
)
logger = logging.getLogger(__name__)

class AnimeRecommenderComparison:
    """Anime recommendation system with multiple model comparison."""
    
    def __init__(self, data_path='data/processed/'):
        """Initialize the recommendation system with data path."""
        self.data_path = data_path
        self.dt_model = None
        self.rf_model = None
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

    # def prepare_features(self, train_df, test_df):
        """Prepare features for model training from the dataset."""
        print("Preparing features for training...")


        
        # Create a copy of anime_features without the rating column to avoid data leakage
        anime_features_copy = self.anime_features.copy()
        
        # # Remove rating-related columns from anime_features that could cause data leakage
        # if 'rating' in anime_features_copy.columns:
        #     anime_features_copy = anime_features_copy.drop('rating', axis=1)

        # Remove rating-related columns from anime_features that could cause data leakage
        if 'rating' in anime_features_copy.columns:
            anime_features_copy = anime_features_copy.rename(columns={'rating':'global_rating'})
        
        # Get basic user features that are appropriate for prediction
        user_features = [
            'user_mean_rating', 
            'user_rating_count', 
            'user_rating_std', 
            'engagement_percentile', 
            'genre_diversity',
            'normalized_rating'
        ]
        
        # Get basic anime features, carefully excluding those derived from user ratings
        anime_features = [
            'episodes', 
            'members', 
            'popularity_percentile',
            'global_rating'
        ]
        
        # Find genre columns - anything that's not one of the basic columns and not a type
        genre_columns = [col for col in self.anime_features.columns 
                         if col not in ['anime_id', 'name', 'episodes', 'rating', 'members',
                                      'rating_count', 'user_avg_rating', 'user_median_rating', 
                                      'user_rating_std', 'popularity_percentile'] 
                         and not col.startswith('type_')]
        
        # Find type columns
        type_columns = [col for col in self.anime_features.columns if col.startswith('type_')]


       
        
        # Merge ratings with anime features without using suffixes
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
        
        # # Remove normalized_rating if it exists (derived from target)
        # if 'normalized_rating' in train_features.columns:
        #     train_features = train_features.drop('normalized_rating', axis=1)
        #     test_features = test_features.drop('normalized_rating', axis=1)
            
        # # Handle top_genres if it exists
        # if 'top_genres' in train_features.columns:
        #     print("Removing top_genres column to avoid potential leakage")
        #     train_features = train_features.drop('top_genres', axis=1)
        #     test_features = test_features.drop('top_genres', axis=1)


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
            genre_df = genre_df.add_prefix('genre_')
            genre_test_df = genre_test_df.add_prefix('genre_')
            
            # Drop the original top_genres column
            train_features = train_features.drop('top_genres', axis=1)
            test_features = test_features.drop('top_genres', axis=1)
            
            # Concatenate the one-hot encoded genres with the original features
            train_features = pd.concat([train_features.reset_index(drop=True), genre_df], axis=1)
            test_features = pd.concat([test_features.reset_index(drop=True), genre_test_df], axis=1)
            
        
        # Handle missing values
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)
        
        # Combine all validated features
        all_features = []
        
        # Only add features that exist in both dataframes
        for feature in user_features + anime_features:
            if feature in train_features.columns:
                all_features.append(feature)
                
        # Add genre and type columns
        for feature in genre_columns + type_columns:
            if feature in train_features.columns:
                all_features.append(feature)
        
        print("***************************************************************************")
        # print(f"Using {len(all_features)} features for training")
        # print(all_features)
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
        # Uncomment to log all features if needed
        # print(all_features)
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

    def train_models(self, X_train, y_train):
        """Train and compare Decision Tree and Random Forest models."""
        print("Training both Decision Tree and Random Forest models...")
        
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
            
        # ====== Decision Tree Model ======
        start_time = time.time()
        print("\nTraining Decision Tree model...")
        
        # Parameter grid for decision tree
        dt_param_grid = {
            'max_depth': [10, 15, 20],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [4, 8]
        }
        
        dt_base = DecisionTreeRegressor(random_state=42)
        
        # Grid search for Decision Tree
        dt_grid = GridSearchCV(
            estimator=dt_base,
            param_grid=dt_param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        dt_grid.fit(X_sample, y_sample)
        
        # Get best parameters
        dt_best_params = dt_grid.best_params_
        print(f"Best Decision Tree parameters: {dt_best_params}")
        
        # Train on full dataset with best parameters
        self.dt_model = DecisionTreeRegressor(random_state=42, **dt_best_params)
        self.dt_model.fit(X_train, y_train)
        
        print(f"Decision Tree trained in {time.time() - start_time:.2f} seconds")
        
        # ====== Random Forest Model ======
        start_time = time.time()
        print("\nTraining Random Forest model...")
        
        # Parameter grid for random forest
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [10, 20]
        }
        
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Grid search for Random Forest
        rf_grid = GridSearchCV(
            estimator=rf_base,
            param_grid=rf_param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        rf_grid.fit(X_sample, y_sample)
        
        # Get best parameters
        rf_best_params = rf_grid.best_params_
        print(f"Best Random Forest parameters: {rf_best_params}")
        
        # Train on full dataset with best parameters
        self.rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_best_params)
        self.rf_model.fit(X_train, y_train)
        
        print(f"Random Forest trained in {time.time() - start_time:.2f} seconds")
        
        # Save models
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/decision_tree_regressors.pkl', 'wb') as f:
                pickle.dump(self.dt_model, f)
            with open('models/random_forest_model_regressor.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
            print("Models saved successfully")
        except Exception as e:
            print(f"Warning: Could not save models: {str(e)}")
            
        return self.dt_model, self.rf_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate and compare both models on test data."""
        if self.dt_model is None or self.rf_model is None:
            print("Error: Models have not been trained. Call train_models() first.")
            return None
            
        print("\n===== MODEL COMPARISON =====")
        results = {}
        
        # Decision Tree evaluation
        print("\nEvaluating Decision Tree model...")
        dt_pred = self.dt_model.predict(X_test)
        
        # Round and clip predictions for discrete evaluation
        dt_pred_rounded = np.round(np.clip(dt_pred, 1, 10))
        
        # Calculate metrics
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
        dt_mae = mean_absolute_error(y_test, dt_pred)
        dt_r2 = r2_score(y_test, dt_pred)
        dt_accuracy = np.mean(dt_pred_rounded == np.round(y_test))
        dt_within_one = np.mean(abs(dt_pred_rounded - np.round(y_test)) <= 1)
        
        print(f"Decision Tree Results:")
        print(f"RMSE: {dt_rmse:.4f}")
        print(f"MAE: {dt_mae:.4f}")
        print(f"R²: {dt_r2:.4f}")
        print(f"Accuracy (rounded): {dt_accuracy:.4f}")
        print(f"Within 1 point: {dt_within_one:.4f}")
        
        results['decision_tree'] = {
            'rmse': dt_rmse,
            'mae': dt_mae,
            'r2': dt_r2,
            'accuracy': dt_accuracy,
            'within_one': dt_within_one
        }
        
        # Decision Tree feature importance
        dt_importances = self.dt_model.feature_importances_
        dt_importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': dt_importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features (Decision Tree):")
        for i, (feature, importance) in enumerate(
            zip(dt_importance_df['Feature'].head(10), 
                dt_importance_df['Importance'].head(10))):
            print(f"{i+1}. {feature}: {importance:.4f}")
            
        # Random Forest evaluation
        print("\nEvaluating Random Forest model...")
        rf_pred = self.rf_model.predict(X_test)
        
        # Round and clip predictions for discrete evaluation
        rf_pred_rounded = np.round(np.clip(rf_pred, 1, 10))
        
        # Calculate metrics
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_accuracy = np.mean(rf_pred_rounded == np.round(y_test))
        rf_within_one = np.mean(abs(rf_pred_rounded - np.round(y_test)) <= 1)
        
        print(f"Random Forest Results:")
        print(f"RMSE: {rf_rmse:.4f}")
        print(f"MAE: {rf_mae:.4f}")
        print(f"R²: {rf_r2:.4f}")
        print(f"Accuracy (rounded): {rf_accuracy:.4f}")
        print(f"Within 1 point: {rf_within_one:.4f}")
        
        results['random_forest'] = {
            'rmse': rf_rmse,
            'mae': rf_mae,
            'r2': rf_r2,
            'accuracy': rf_accuracy,
            'within_one': rf_within_one
        }
        
        # Random Forest feature importance
        rf_importances = self.rf_model.feature_importances_
        rf_importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': rf_importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features (Random Forest):")
        for i, (feature, importance) in enumerate(
            zip(rf_importance_df['Feature'].head(10), 
                rf_importance_df['Importance'].head(10))):
            print(f"{i+1}. {feature}: {importance:.4f}")
            
        # Print model comparison summary
        print("\n===== MODEL COMPARISON SUMMARY =====")
        print(f"{'Metric':<20} {'Decision Tree':<15} {'Random Forest':<15} {'Difference':<15}")
        print("-" * 65)
        print(f"{'RMSE':<20} {dt_rmse:<15.4f} {rf_rmse:<15.4f} {abs(dt_rmse - rf_rmse):<15.4f}")
        print(f"{'MAE':<20} {dt_mae:<15.4f} {rf_mae:<15.4f} {abs(dt_mae - rf_mae):<15.4f}")
        print(f"{'R²':<20} {dt_r2:<15.4f} {rf_r2:<15.4f} {abs(dt_r2 - rf_r2):<15.4f}")
        print(f"{'Accuracy':<20} {dt_accuracy:<15.4f} {rf_accuracy:<15.4f} {abs(dt_accuracy - rf_accuracy):<15.4f}")
        print(f"{'Within 1 point':<20} {dt_within_one:<15.4f} {rf_within_one:<15.4f} {abs(dt_within_one - rf_within_one):<15.4f}")
        
        # Compare feature importance between models
        print("\n===== FEATURE IMPORTANCE COMPARISON =====")
        print("Top 5 features comparison:")
        print(f"{'Rank':<5} {'Decision Tree':<25} {'Random Forest':<25}")
        print("-" * 55)
        for i in range(5):
            dt_feature = dt_importance_df['Feature'].iloc[i]
            dt_importance = dt_importance_df['Importance'].iloc[i]
            rf_feature = rf_importance_df['Feature'].iloc[i]
            rf_importance = rf_importance_df['Importance'].iloc[i]
            print(f"{i+1:<5} {dt_feature:<20} ({dt_importance:.4f}) {rf_feature:<20} ({rf_importance:.4f})")
        
        return results

    def generate_error_distribution(self, X_test, y_test):
        """Generate and visualize error distributions for both models."""
        print("\nGenerating error distribution analysis...")
        
        # Decision Tree predictions
        dt_pred = self.dt_model.predict(X_test)
        dt_errors = y_test - dt_pred
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(X_test)
        rf_errors = y_test - rf_pred
        
        # Calculate error statistics
        print("\nError Statistics:")
        print(f"{'Metric':<20} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 50)
        print(f"{'Mean Error':<20} {dt_errors.mean():<15.4f} {rf_errors.mean():<15.4f}")
        print(f"{'Error Std Dev':<20} {dt_errors.std():<15.4f} {rf_errors.std():<15.4f}")
        print(f"{'Min Error':<20} {dt_errors.min():<15.4f} {rf_errors.min():<15.4f}")
        print(f"{'Max Error':<20} {dt_errors.max():<15.4f} {rf_errors.max():<15.4f}")
        
        # Analyze errors by true rating
        rating_groups = pd.DataFrame({
            'true_rating': y_test,
            'dt_error': np.abs(dt_errors),
            'rf_error': np.abs(rf_errors)
        }).groupby('true_rating')
        
        error_by_rating = rating_groups.mean()
        
        print("\nMean Absolute Error by True Rating:")
        print(error_by_rating)
        
        return {
            'dt_errors': dt_errors,
            'rf_errors': rf_errors,
            'error_by_rating': error_by_rating
        }

    def analyze_feature_correlations(self, X_train):
        """Analyze and visualize correlations between input features."""
        print("\n===== FEATURE CORRELATION ANALYSIS =====")
        
        # Compute the correlation matrix
        correlation_matrix = X_train.corr()
        
        # Save full correlation matrix to CSV
        os.makedirs('analysis', exist_ok=True)
        correlation_matrix.to_csv('analysis/feature_correlation_matrix.csv')
        print(f"Full correlation matrix saved to 'analysis/feature_correlation_matrix.csv'")
        
        # Find highly correlated feature pairs (absolute correlation > 0.7)
        high_corr_threshold = 0.7
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        # Sort by absolute correlation value (descending)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Print highly correlated feature pairs
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (|correlation| > {high_corr_threshold}):")
            print(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<10}")
            print("-" * 70)
            
            for feature1, feature2, corr in high_corr_pairs[:20]:  # Show top 20
                print(f"{feature1:<30} {feature2:<30} {corr:>10.4f}")
                
            print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs in total.")
        else:
            print(f"No feature pairs with correlation > {high_corr_threshold} found.")
        
        # Create a correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # If there are many features, show only the most important ones
        if len(self.feature_columns) > 20:
            # Get top 20 features by importance from both models
            dt_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            rf_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Combine top features from both models
            top_features = set(dt_importances['Feature'].head(15).tolist() + 
                              rf_importances['Feature'].head(15).tolist())
            
            # Create a reduced correlation matrix
            reduced_corr = correlation_matrix.loc[top_features, top_features]
            
            # Plot the reduced heatmap
            import seaborn as sns
            sns.heatmap(reduced_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Heatmap of Top Features')
            plt.tight_layout()
            plt.savefig('analysis/top_features_correlation_heatmap.png')
            print(f"Correlation heatmap of top features saved to 'analysis/top_features_correlation_heatmap.png'")
        else:
            # Plot full heatmap if there aren't too many features
            import seaborn as sns
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('analysis/feature_correlation_heatmap.png')
            print(f"Full correlation heatmap saved to 'analysis/feature_correlation_heatmap.png'")
        
        # Group features by type for focused correlation analysis
        user_features = [f for f in self.feature_columns if f in [
            'user_mean_rating', 'user_rating_count', 'user_rating_std', 
            'engagement_percentile', 'genre_diversity'
        ]]
        
        anime_features = [f for f in self.feature_columns if f in [
            'episodes', 'members', 'popularity_percentile', 'global_rating'
        ]]
        
        genre_features = [f for f in self.feature_columns if 
                         (f not in user_features and f not in anime_features and 
                          not f.startswith('type_') and not f.startswith('user_genre_'))]
        
        # Print correlation summaries by feature groups
        print("\n=== Correlation between main user features ===")
        if len(user_features) > 1:
            user_corr = correlation_matrix.loc[user_features, user_features]
            print(user_corr)
        else:
            print("Not enough user features for correlation analysis")
            
        print("\n=== Correlation between main anime features ===")
        if len(anime_features) > 1:
            anime_corr = correlation_matrix.loc[anime_features, anime_features]
            print(anime_corr)
        else:
            print("Not enough anime features for correlation analysis")
        
        # Check correlation of features with target (if available)
        try:
            if hasattr(self, 'y_train') and self.y_train is not None:
                # Create a dataframe with features and target
                target_corr_df = X_train.copy()
                target_corr_df['rating'] = self.y_train.values
                
                # Compute correlation with target
                target_corr = target_corr_df.corr()['rating'].drop('rating')
                target_corr = target_corr.sort_values(ascending=False)
                
                print("\n=== Feature correlation with target (rating) ===")
                print(target_corr.head(15))
                print("\n(Showing top 15 positively correlated features)")
                
                # Save target correlations
                target_corr.to_csv('analysis/feature_target_correlation.csv')
                print(f"Target correlations saved to 'analysis/feature_target_correlation.csv'")
        except Exception as e:
            print(f"Could not compute correlation with target: {str(e)}")
        
        print("\nFeature correlation analysis complete.")
        return correlation_matrix


def main():
    """Main function to compare Decision Tree and Random Forest models."""
    print("Starting Anime Recommendation System Model Comparison...")
    
    # Create an instance of the recommender
    recommender = AnimeRecommenderComparison()
    
    try:
        # Load data
        print("\n==== LOADING DATA ====")
        train_data, test_data = recommender.load_data()
        
        # Prepare features
        print("\n==== PREPARING FEATURES ====")
        X_train, y_train, X_test, y_test = recommender.prepare_features(train_data, test_data)
        
        # Store y_train for correlation analysis
        recommender.y_train = y_train
        
        # Feature correlation analysis
        print("\n==== FEATURE CORRELATION ANALYSIS ====")
        recommender.analyze_feature_correlations(X_train)
        
        # Train both models
        print("\n==== TRAINING MODELS ====")
        dt_model, rf_model = recommender.train_models(X_train, y_train)
        
        # Evaluate and compare models
        print("\n==== EVALUATING MODELS ====")
        results = recommender.evaluate_models(X_test, y_test)
        
        # Generate error distributions
        print("\n==== ERROR ANALYSIS ====")
        error_analysis = recommender.generate_error_distribution(X_test, y_test)
        
        print("\nModel comparison completed successfully!")
        
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Model comparison failed to complete.")
    
if __name__ == "__main__":
    main()