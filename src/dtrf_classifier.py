import pandas as pd
import numpy as np
import os
import logging
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    filename="anime_recommender_classifier.log"
)
logger = logging.getLogger(__name__)

class AnimeRecommenderClassifierComparison:
    """Anime recommendation system with multiple classification model comparison."""
    
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
        
        # Convert continuous ratings to discrete classes
        # For classification, we'll treat each discrete rating value as a class
        y_train = train_features['rating'].round().astype(int)
        
        X_test = test_features[all_features]
        y_test = test_features['rating'].round().astype(int)
        
        print(f"Features prepared. X_train shape: {X_train.shape}")
        logger.info(f"Features prepared. Using {len(all_features)} features.")
        
        # Check class distribution
        print("\nClass distribution in training set:")
        train_class_dist = y_train.value_counts().sort_index()
        for rating, count in train_class_dist.items():
            percentage = count / len(y_train) * 100
            print(f"Rating {rating}: {count} samples ({percentage:.2f}%)")
        
        return X_train, y_train, X_test, y_test

    def train_models(self, X_train, y_train):
        """Train and compare Decision Tree and Random Forest classifier models."""
        print("Training both Decision Tree and Random Forest classifier models...")
        
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
        print("\nTraining Decision Tree classifier model...")
        
        # Parameter grid for decision tree
        dt_param_grid = {
            'max_depth': [10, 15, 20],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [4, 8]
        }
        
        dt_base = DecisionTreeClassifier(random_state=42)
        
        # Grid search for Decision Tree
        dt_grid = GridSearchCV(
            estimator=dt_base,
            param_grid=dt_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        dt_grid.fit(X_sample, y_sample)
        
        # Get best parameters
        dt_best_params = dt_grid.best_params_
        print(f"Best Decision Tree parameters: {dt_best_params}")
        
        # Train on full dataset with best parameters
        self.dt_model = DecisionTreeClassifier(random_state=42, **dt_best_params)
        self.dt_model.fit(X_train, y_train)
        
        print(f"Decision Tree trained in {time.time() - start_time:.2f} seconds")
        
        # ====== Random Forest Model ======
        start_time = time.time()
        print("\nTraining Random Forest classifier model...")
        
        # Parameter grid for random forest
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [10, 20]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search for Random Forest
        rf_grid = GridSearchCV(
            estimator=rf_base,
            param_grid=rf_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(X_sample, y_sample)
        
        # Get best parameters
        rf_best_params = rf_grid.best_params_
        print(f"Best Random Forest parameters: {rf_best_params}")
        
        # Train on full dataset with best parameters
        self.rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_best_params)
        self.rf_model.fit(X_train, y_train)
        
        print(f"Random Forest trained in {time.time() - start_time:.2f} seconds")
        
        # Save models
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/decision_tree_classifier.pkl', 'wb') as f:
                pickle.dump(self.dt_model, f)
            with open('models/random_forest_classifier.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
            print("Models saved successfully")
        except Exception as e:
            print(f"Warning: Could not save models: {str(e)}")
            
        return self.dt_model, self.rf_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate and compare both classification models on test data."""
        if self.dt_model is None or self.rf_model is None:
            print("Error: Models have not been trained. Call train_models() first.")
            return None
            
        print("\n===== MODEL COMPARISON =====")
        results = {}
        
        # Define the class labels (possible ratings)
        labels = sorted(y_test.unique())
        
        # Decision Tree evaluation
        print("\nEvaluating Decision Tree classifier model...")
        dt_pred = self.dt_model.predict(X_test)
        
        # Calculate standard classification metrics
        dt_accuracy = accuracy_score(y_test, dt_pred)
        dt_precision = precision_score(y_test, dt_pred, labels=labels, average='weighted', zero_division=0)
        dt_recall = recall_score(y_test, dt_pred, labels=labels, average='weighted', zero_division=0)
        dt_f1 = f1_score(y_test, dt_pred, labels=labels, average='weighted', zero_division=0)
        
        # Calculate within-one accuracy (predictions within 1 rating point)
        dt_within_one = np.mean(abs(dt_pred - y_test) <= 1)
        
        print(f"Decision Tree Results:")
        print(f"Accuracy: {dt_accuracy:.4f}")
        print(f"Precision (weighted): {dt_precision:.4f}")
        print(f"Recall (weighted): {dt_recall:.4f}")
        print(f"F1 Score (weighted): {dt_f1:.4f}")
        print(f"Within 1 point: {dt_within_one:.4f}")
        
        # Build confusion matrix for Decision Tree
        dt_cm = confusion_matrix(y_test, dt_pred, labels=labels)
        
        results['decision_tree'] = {
            'accuracy': dt_accuracy,
            'precision': dt_precision,
            'recall': dt_recall,
            'f1': dt_f1,
            'within_one': dt_within_one,
            'confusion_matrix': dt_cm
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
        print("\nEvaluating Random Forest classifier model...")
        rf_pred = self.rf_model.predict(X_test)
        
        # Calculate standard classification metrics
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred, labels=labels, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, rf_pred, labels=labels, average='weighted', zero_division=0)
        rf_f1 = f1_score(y_test, rf_pred, labels=labels, average='weighted', zero_division=0)
        
        # Calculate within-one accuracy (predictions within 1 rating point)
        rf_within_one = np.mean(abs(rf_pred - y_test) <= 1)
        
        print(f"Random Forest Results:")
        print(f"Accuracy: {rf_accuracy:.4f}")
        print(f"Precision (weighted): {rf_precision:.4f}")
        print(f"Recall (weighted): {rf_recall:.4f}")
        print(f"F1 Score (weighted): {rf_f1:.4f}")
        print(f"Within 1 point: {rf_within_one:.4f}")
        
        # Build confusion matrix for Random Forest
        rf_cm = confusion_matrix(y_test, rf_pred, labels=labels)
        
        results['random_forest'] = {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1': rf_f1,
            'within_one': rf_within_one,
            'confusion_matrix': rf_cm
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
        print(f"{'Accuracy':<20} {dt_accuracy:<15.4f} {rf_accuracy:<15.4f} {abs(dt_accuracy - rf_accuracy):<15.4f}")
        print(f"{'Precision':<20} {dt_precision:<15.4f} {rf_precision:<15.4f} {abs(dt_precision - rf_precision):<15.4f}")
        print(f"{'Recall':<20} {dt_recall:<15.4f} {rf_recall:<15.4f} {abs(dt_recall - rf_recall):<15.4f}")
        print(f"{'F1 Score':<20} {dt_f1:<15.4f} {rf_f1:<15.4f} {abs(dt_f1 - rf_f1):<15.4f}")
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

    def analyze_class_performance(self, X_test, y_test):
        """Analyze model performance by class (rating value)."""
        print("\nAnalyzing performance by rating class...")
        
        # Decision Tree predictions
        dt_pred = self.dt_model.predict(X_test)
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(X_test)
        
        # Create dataframe with true and predicted values
        results_df = pd.DataFrame({
            'true_rating': y_test,
            'dt_pred': dt_pred,
            'rf_pred': rf_pred
        })
        
        # Calculate per-class accuracy for each model
        dt_class_accuracy = {}
        rf_class_accuracy = {}
        
        unique_ratings = sorted(y_test.unique())
        
        print("\nPer-Class Accuracy:")
        print(f"{'Rating':<10} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 40)
        
        for rating in unique_ratings:
            class_mask = (y_test == rating)
            dt_class_acc = accuracy_score(y_test[class_mask], dt_pred[class_mask])
            rf_class_acc = accuracy_score(y_test[class_mask], rf_pred[class_mask])
            
            dt_class_accuracy[rating] = dt_class_acc
            rf_class_accuracy[rating] = rf_class_acc
            
            print(f"{rating:<10} {dt_class_acc:<15.4f} {rf_class_acc:<15.4f}")
        
        # Calculate "within 1" accuracy by class
        dt_within_one = {}
        rf_within_one = {}
        
        print("\nPer-Class 'Within 1' Accuracy:")
        print(f"{'Rating':<10} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 40)
        
        for rating in unique_ratings:
            class_mask = (y_test == rating)
            dt_within_one[rating] = np.mean(abs(dt_pred[class_mask] - rating) <= 1)
            rf_within_one[rating] = np.mean(abs(rf_pred[class_mask] - rating) <= 1)
            
            print(f"{rating:<10} {dt_within_one[rating]:<15.4f} {rf_within_one[rating]:<15.4f}")
        
        return {
            'dt_class_accuracy': dt_class_accuracy,
            'rf_class_accuracy': rf_class_accuracy,
            'dt_within_one': dt_within_one,
            'rf_within_one': rf_within_one
        }

def main():
    """Main function to compare Decision Tree and Random Forest classification models."""
    print("Starting Anime Recommendation System Classification Model Comparison...")
    
    # Create an instance of the recommender
    recommender = AnimeRecommenderClassifierComparison()
    
    try:
        # Load data
        print("\n==== LOADING DATA ====")
        train_data, test_data = recommender.load_data()
        
        # Prepare features
        print("\n==== PREPARING FEATURES ====")
        X_train, y_train, X_test, y_test = recommender.prepare_features(train_data, test_data)
        
        # Train both models
        print("\n==== TRAINING MODELS ====")
        dt_model, rf_model = recommender.train_models(X_train, y_train)
        
        # Evaluate and compare models
        print("\n==== EVALUATING MODELS ====")
        results = recommender.evaluate_models(X_test, y_test)
        
        # Analyze class-specific performance
        print("\n==== CLASS PERFORMANCE ANALYSIS ====")
        class_analysis = recommender.analyze_class_performance(X_test, y_test)
        
        print("\nClassification model comparison completed successfully!")
        
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Model comparison failed to complete.")
    
if __name__ == "__main__":
    main() 