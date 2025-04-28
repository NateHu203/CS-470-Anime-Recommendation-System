import pandas as pd
import numpy as np
import os
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix)
import joblib
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="model_comparison.log"
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive evaluation for different anime recommendation models."""
    
    def __init__(self, data_path='data/processed/'):
        """Initialize the evaluator with data path."""
        self.data_path = data_path
        self.feature_columns = None
        self.models = {}
        self.model_types = {}  # To track if model is classifier or regressor
        self.anime_features = None
        self.anime_dict = {}  # For quick lookups
        
    def load_data(self):
        """Load the necessary data files for evaluation."""
        print("Loading data files...")
        
        try:
            # Load the preprocessed data files
            self.anime_features = pd.read_csv(os.path.join("..", self.data_path, 'anime_features_normalized_new.csv'))
            
            # Load test set (we only need test set for evaluation)
            test_data = pd.read_csv(os.path.join("..", self.data_path, 'rating_test_v2_new.csv'))
            
            # Create a dictionary for quick anime lookups
            self.anime_dict = self.anime_features.set_index('anime_id').to_dict('index')
            
            print(f"Data loaded successfully. Test set: {len(test_data)} samples")
            logger.info(f"Data loaded. Test set: {len(test_data)} samples")
            
            return test_data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, test_df):
        """Prepare features for model evaluation."""
        print("Preparing features for evaluation...")
        
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
        test_features = pd.merge(
            test_df, 
            anime_features_copy,
            on='anime_id', 
            how='left'
        )
        
        # Remove normalized_rating if it exists (derived from target)
        if 'normalized_rating' in test_features.columns:
            print("Removing normalized_rating column to avoid data leakage")
            test_features = test_features.drop('normalized_rating', axis=1)
        
        # Load expected genre features before processing top_genres
        self.extract_expected_features()
        
        # Process top_genres column using the expected genre features from models
        if 'top_genres' in test_features.columns:
            print("Processing top_genres column for model compatibility...")
            
            # Convert string representations of lists to actual lists
            test_features['top_genres'] = test_features['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
            
            # Create a DataFrame with all expected genre features, initialized to 0
            # Ensure we include ALL potential genre features used by any model
            genre_test_df = pd.DataFrame(0, index=test_features.index, columns=self.expected_genre_features)
            
            # Fill in 1s for genres that exist in the top_genres
            for idx, genres in enumerate(test_features['top_genres']):
                if isinstance(genres, list):  # Make sure it's a list
                    for genre in genres:
                        if not genre:  # Skip empty strings
                            continue
                        genre_key = f"user_genre_{genre}"
                        if genre_key in self.expected_genre_features:
                            genre_test_df.loc[idx, genre_key] = 1
            
            print(f"Created {len(self.expected_genre_features)} genre features aligned with model expectations")
                
            # Drop the original top_genres column
            test_features = test_features.drop('top_genres', axis=1)
            
            # Concatenate the genre features with the original features
            test_features = pd.concat([test_features.reset_index(drop=True), genre_test_df.reset_index(drop=True)], axis=1)
        else:
            # If top_genres column doesn't exist, still need to create the expected user_genre_* features
            print(f"No top_genres column found. Creating {len(self.expected_genre_features)} empty genre features...")
            genre_test_df = pd.DataFrame(0, index=test_features.index, columns=self.expected_genre_features)
            test_features = pd.concat([test_features.reset_index(drop=True), genre_test_df.reset_index(drop=True)], axis=1)
        
        # Handle missing values
        test_features = test_features.fillna(0)
        
        # Collect all features to use - starting with what's in the data
        all_features = []
        
        # Add user features
        for feature in user_features:
            if feature in test_features.columns:
                all_features.append(feature)
        
        # Add anime features
        for feature in anime_features:
            if feature in test_features.columns:
                all_features.append(feature)
        
        # Add genre columns
        for feature in genre_columns:
            if feature in test_features.columns:
                all_features.append(feature)
        
        # Add type columns
        for feature in type_columns:
            if feature in test_features.columns:
                all_features.append(feature)
        
        # Add expected genre features (all of them)
        for feature in self.expected_genre_features:
            if feature not in all_features and feature in test_features.columns:
                all_features.append(feature)
        
        print(f"Using {len(all_features)} features for evaluation")
        logger.info(f"Features for evaluation: {all_features}")
        
        self.feature_columns = all_features
        
        # Create feature matrices and target vectors
        X_test = test_features[all_features]
        y_test = test_features['rating']
        
        print(f"Features prepared. X_test shape: {X_test.shape}")
        
        return X_test, y_test

    def load_models(self):
        """Load the 6 models for comparison."""
        print("Loading all models for comparison...")
        models_dir = 'models'
        
        # 1. Load Decision Tree Classifier
        dt_classifier_path = os.path.join(models_dir, 'decision_tree_classifier.pkl')
        if os.path.exists(dt_classifier_path):
            print("Loading Decision Tree Classifier...")
            try:
                with open(dt_classifier_path, 'rb') as f:
                    self.models['dt_classifier'] = pickle.load(f)
                self.model_types['dt_classifier'] = 'classifier'
                # Store feature names if available
                if hasattr(self.models['dt_classifier'], 'feature_names_in_'):
                    self.models['dt_classifier_features'] = self.models['dt_classifier'].feature_names_in_.tolist()
                print("Decision Tree Classifier loaded successfully")
            except Exception as e:
                print(f"Error loading Decision Tree Classifier: {e}")
        else:
            print(f"Decision Tree Classifier not found at {dt_classifier_path}")
        
        # 2. Load Random Forest Classifier
        rf_classifier_path = os.path.join(models_dir, 'random_forest_classifier.pkl')
        if os.path.exists(rf_classifier_path):
            print("Loading Random Forest Classifier...")
            try:
                with open(rf_classifier_path, 'rb') as f:
                    self.models['rf_classifier'] = pickle.load(f)
                self.model_types['rf_classifier'] = 'classifier'
                # Store feature names if available
                if hasattr(self.models['rf_classifier'], 'feature_names_in_'):
                    self.models['rf_classifier_features'] = self.models['rf_classifier'].feature_names_in_.tolist()
                print("Random Forest Classifier loaded successfully")
            except Exception as e:
                print(f"Error loading Random Forest Classifier: {e}")
        else:
            print(f"Random Forest Classifier not found at {rf_classifier_path}")
        
        # 3. Load KNN Classifier
        knn_classifier_path = os.path.join(models_dir, 'anime_knn_classifier.joblib')
        if os.path.exists(knn_classifier_path):
            print("Loading KNN Classifier...")
            try:
                knn_data = joblib.load(knn_classifier_path)
                self.models['knn_classifier'] = knn_data['model']
                self.model_types['knn_classifier'] = 'classifier'
                # Store feature names if available
                if 'feature_columns' in knn_data:
                    self.models['knn_classifier_features'] = knn_data['feature_columns']
                # Store PCA and scaler for preprocessing
                self.models['knn_classifier_pca'] = knn_data.get('pca')
                self.models['knn_classifier_scaler'] = knn_data.get('scaler')
                self.models['knn_classifier_n_components'] = knn_data.get('n_components', 20)
                print("KNN Classifier loaded successfully")
            except Exception as e:
                print(f"Error loading KNN Classifier: {e}")
        else:
            print(f"KNN Classifier not found at {knn_classifier_path}")
        
        # 4. Load Decision Tree Regressor
        dt_regressor_path = os.path.join(models_dir, 'decision_tree_regressors.pkl')
        if os.path.exists(dt_regressor_path):
            print("Loading Decision Tree Regressor...")
            try:
                with open(dt_regressor_path, 'rb') as f:
                    self.models['dt_regressor'] = pickle.load(f)
                self.model_types['dt_regressor'] = 'regressor'
                # Store feature names if available
                if hasattr(self.models['dt_regressor'], 'feature_names_in_'):
                    self.models['dt_regressor_features'] = self.models['dt_regressor'].feature_names_in_.tolist()
                print("Decision Tree Regressor loaded successfully")
            except Exception as e:
                print(f"Error loading Decision Tree Regressor: {e}")
        else:
            print(f"Decision Tree Regressor not found at {dt_regressor_path}")
        
        # 5. Load Random Forest Regressor
        rf_regressor_path = os.path.join(models_dir, 'random_forest_model_regressor.pkl')
        if os.path.exists(rf_regressor_path):
            print("Loading Random Forest Regressor...")
            try:
                with open(rf_regressor_path, 'rb') as f:
                    self.models['rf_regressor'] = pickle.load(f)
                self.model_types['rf_regressor'] = 'regressor'
                # Store feature names if available
                if hasattr(self.models['rf_regressor'], 'feature_names_in_'):
                    self.models['rf_regressor_features'] = self.models['rf_regressor'].feature_names_in_.tolist()
                print("Random Forest Regressor loaded successfully")
            except Exception as e:
                print(f"Error loading Random Forest Regressor: {e}")
        else:
            print(f"Random Forest Regressor not found at {rf_regressor_path}")
        
        # 6. Load KNN Regressor
        knn_regressor_path = os.path.join(models_dir, 'anime_knn_model_dtrf_style.joblib')
        if os.path.exists(knn_regressor_path):
            print("Loading KNN Regressor...")
            try:
                knn_data = joblib.load(knn_regressor_path)
                self.models['knn_regressor'] = knn_data['model']
                self.model_types['knn_regressor'] = 'regressor'
                # Store feature names if available
                if 'feature_columns' in knn_data:
                    self.models['knn_regressor_features'] = knn_data['feature_columns']
                # Store PCA and scaler for preprocessing
                self.models['knn_regressor_pca'] = knn_data.get('pca')
                self.models['knn_regressor_scaler'] = knn_data.get('scaler')
                self.models['knn_regressor_n_components'] = knn_data.get('n_components', 20)
                print("KNN Regressor loaded successfully")
            except Exception as e:
                print(f"Error loading KNN Regressor: {e}")
        else:
            print(f"KNN Regressor not found at {knn_regressor_path}")
        
        # Extract all expected features from the models
        self.extract_expected_features()
        
        print(f"Loaded {len(self.models)} models for comparison")
        return self.models
    
    def extract_expected_features(self):
        """Extract and merge all expected feature names from the loaded models."""
        # Initialize a set to collect all expected genre features
        self.expected_genre_features = set()
        
        # Collect feature names from all models
        for model_name in self.models.keys():
            if model_name.endswith('_features') and isinstance(self.models[model_name], list):
                # Extract genre features (those starting with 'user_genre_')
                genre_features = [f for f in self.models[model_name] if f.startswith('user_genre_')]
                self.expected_genre_features.update(genre_features)
        
        # Convert to sorted list for consistent ordering
        self.expected_genre_features = sorted(list(self.expected_genre_features))
        
        # Print detected genre features
        if self.expected_genre_features:
            print(f"Detected {len(self.expected_genre_features)} expected genre features from models")
            logger.info(f"Expected genre features: {self.expected_genre_features}")
        else:
            print("Warning: No genre features detected from models. Feature alignment may not work.")
            # Provide a fallback list of common genres in case we couldn't extract from models
            self.expected_genre_features = [
                'user_genre_Action', 'user_genre_Adventure', 'user_genre_Comedy', 'user_genre_Drama',
                'user_genre_Fantasy', 'user_genre_Horror', 'user_genre_Mystery', 'user_genre_Romance',
                'user_genre_Sci-Fi', 'user_genre_Slice of Life', 'user_genre_Sports', 'user_genre_Supernatural',
                'user_genre_Music', 'user_genre_Mecha', 'user_genre_Military', 'user_genre_Magic', 
                'user_genre_School', 'user_genre_Historical', 'user_genre_Shounen', 'user_genre_Shoujo',
                'user_genre_Cars', 'user_genre_Dementia', 'user_genre_Demons', 'user_genre_Ecchi',
                'user_genre_Game', 'user_genre_Harem', 'user_genre_Hentai', 'user_genre_Josei',
                'user_genre_Kids', 'user_genre_Martial Arts', 'user_genre_Parody', 'user_genre_Police',
                'user_genre_Psychological', 'user_genre_Samurai', 'user_genre_Seinen', 'user_genre_Space',
                'user_genre_Super Power', 'user_genre_Thriller', 'user_genre_Vampire', 'user_genre_Yaoi',
                'user_genre_Yuri', 'user_genre_Comedy', 'user_genre_Cartoon'
            ]
            print(f"Using fallback list of {len(self.expected_genre_features)} common genre features")
            logger.info(f"Fallback genre features: {self.expected_genre_features}")

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all loaded models using consistent metrics."""
        print("\n===== EVALUATING ALL MODELS =====")
        if not self.models:
            print("No models loaded. Please call load_models() first.")
            return None
        
        results = {}
        
        for model_name, model in self.models.items():
            # Skip non-model entries (e.g., pca, scaler components, features lists)
            if (model_name.endswith('_pca') or model_name.endswith('_scaler') or 
                model_name.endswith('_n_components') or model_name.endswith('_features')):
                continue
            
            print(f"\nEvaluating {model_name}...")
            model_type = self.model_types[model_name]
            
            # Make predictions
            try:
                # Check if we have specific feature names for this model
                model_features = None
                if f"{model_name}_features" in self.models:
                    model_features = self.models[f"{model_name}_features"]
                    print(f"Using {len(model_features)} model-specific features for {model_name}")
                    
                    # Create a DataFrame with features in the EXACT order they were during training
                    X_test_for_model = pd.DataFrame(index=X_test.index)
                    
                    # Check for missing features
                    missing_features = [f for f in model_features if f not in X_test.columns]
                    if missing_features:
                        print(f"Warning: {len(missing_features)} features required by the model are missing in test data")
                        print(f"Missing features: {missing_features[:5]}..." if len(missing_features) > 5 else f"Missing features: {missing_features}")
                        
                    # Add each feature in the exact order used during training
                    for feature in model_features:
                        if feature in X_test.columns:
                            X_test_for_model[feature] = X_test[feature]
                        else:
                            # Add missing features with zero values
                            X_test_for_model[feature] = 0
                    
                    print(f"Created feature matrix with exact same order as training")
                else:
                    # If we don't have specific feature info, use all features
                    X_test_for_model = X_test
                    print(f"No model-specific features found for {model_name}, using all {X_test.shape[1]} features")
                
                # Special handling for KNN models which may need dimensionality reduction
                if model_name in ['knn_classifier', 'knn_regressor']:
                    # Check if we need to apply PCA and scaling
                    pca = self.models.get(f'{model_name}_pca')
                    scaler = self.models.get(f'{model_name}_scaler')
                    
                    if scaler is not None and pca is not None:
                        print(f"Applying scaling and PCA for {model_name}")
                        X_test_scaled = scaler.transform(X_test_for_model)
                        X_test_pca = pca.transform(X_test_scaled)
                        y_pred = model.predict(X_test_pca)
                    elif scaler is not None:
                        print(f"Applying only scaling for {model_name}")
                        X_test_scaled = scaler.transform(X_test_for_model)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        print(f"WARNING: No PCA/scaler found for {model_name}, trying direct prediction")
                        y_pred = model.predict(X_test_for_model)
                else:
                    y_pred = model.predict(X_test_for_model)
                
                # For regressors, round predictions to integers
                if model_type == 'regressor':
                    y_pred_discrete = np.round(np.clip(y_pred, 1, 10)).astype(int)
                else:  # classifier predictions are already integers
                    y_pred_discrete = y_pred
                
                # Round ground truth for consistent comparison
                y_test_discrete = np.round(y_test).astype(int)
                
                # Calculate common metrics
                
                # 1. Accuracy (exact matches)
                accuracy = accuracy_score(y_test_discrete, y_pred_discrete)
                
                # 2. "Within 1 point" accuracy
                within_one = np.mean(abs(y_pred_discrete - y_test_discrete) <= 1)
                
                # 3. RMSE (for both classifiers and regressors)
                if model_type == 'classifier':
                    # For classifiers, calculate RMSE on the integer predictions
                    rmse = np.sqrt(mean_squared_error(y_test_discrete, y_pred_discrete))
                else:  # regressor
                    # For regressors, calculate RMSE on the raw predictions
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # 4. MAE (for both)
                if model_type == 'classifier':
                    mae = mean_absolute_error(y_test_discrete, y_pred_discrete)
                else:  # regressor
                    mae = mean_absolute_error(y_test, y_pred)
                
                # 5. F1 score (weighted for multi-class)
                try:
                    f1 = f1_score(y_test_discrete, y_pred_discrete, average='weighted')
                    precision = precision_score(y_test_discrete, y_pred_discrete, average='weighted', zero_division=0)
                    recall = recall_score(y_test_discrete, y_pred_discrete, average='weighted', zero_division=0)
                except Exception as e:
                    print(f"Error calculating F1/precision/recall: {e}")
                    f1 = precision = recall = None
                
                # 6. R² score (for regressors only)
                if model_type == 'regressor':
                    r2 = r2_score(y_test, y_pred)
                else:
                    r2 = None
                
                # Store results
                results[model_name] = {
                    'model_type': model_type,
                    'accuracy': accuracy,
                    'within_one': within_one,
                    'rmse': rmse,
                    'mae': mae,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'r2': r2,
                    'predictions': y_pred_discrete
                }
                
                # Print results
                print(f"Model Type: {model_type.upper()}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Within 1 Point: {within_one:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                if f1 is not None:
                    print(f"F1 Score: {f1:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                if r2 is not None:
                    print(f"R² Score: {r2:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e), 'model_type': model_type}
        
        return results
    
    def plot_comparison(self, results):
        """Plot comparison of all models."""
        if not results:
            print("No results to plot.")
            return pd.DataFrame()  # Return empty DataFrame if no results
        
        # Filter out models with errors
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid models to plot.")
            return pd.DataFrame()  # Return empty DataFrame if no valid results
        
        # Extract metrics for plotting
        model_names = list(valid_results.keys())
        metrics = ['accuracy', 'within_one', 'rmse', 'mae', 'f1']
        
        # Create dictionary for each metric
        metric_data = {metric: [] for metric in metrics}
        model_types = []
        
        for model_name in model_names:
            model_types.append(valid_results[model_name]['model_type'])
            
            for metric in metrics:
                if metric in valid_results[model_name] and valid_results[model_name][metric] is not None:
                    metric_data[metric].append(valid_results[model_name][metric])
                else:
                    # Use NaN instead of 0 to better indicate missing data
                    metric_data[metric].append(float('nan'))
        
        # Check if we have data to plot
        if not model_names or all(all(np.isnan(val) for val in metric_data[m]) for m in metrics):
            print("No data to plot.")
            return pd.DataFrame()
        
        # Create figure for metrics where higher is better
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        x = range(len(model_names))
        width = 0.2
        
        # Plot accuracy, within_one, and f1 (higher is better)
        accuracy_bars = plt.bar([i - width for i in x], metric_data['accuracy'], width=width, label='Accuracy', color='blue')
        within_one_bars = plt.bar(x, metric_data['within_one'], width=width, label='Within 1 Point', color='green')
        f1_bars = plt.bar([i + width for i in x], metric_data['f1'], width=width, label='F1 Score', color='purple')
        
        # Add value labels above each bar
        for bars in [accuracy_bars, within_one_bars, f1_bars]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):  # Only label non-NaN values
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylabel('Score (higher is better)')
        plt.title('Model Comparison - Accuracy Metrics')
        plt.legend()
        plt.ylim(0, 1.1)  # Limit y-axis for better visualization
        
        # Create second subplot for metrics where lower is better
        plt.subplot(2, 1, 2)
        
        # Plot RMSE and MAE (lower is better)
        rmse_bars = plt.bar([i - width/2 for i in x], metric_data['rmse'], width=width, label='RMSE', color='red')
        mae_bars = plt.bar([i + width/2 for i in x], metric_data['mae'], width=width, label='MAE', color='orange')
        
        # Add value labels above each bar
        for bars in [rmse_bars, mae_bars]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):  # Only label non-NaN values
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylabel('Error (lower is better)')
        plt.title('Model Comparison - Error Metrics')
        plt.legend()
        
        # Set appropriate y-limit based on data
        max_error = max([
            max([val for val in metric_data['rmse'] if not np.isnan(val)], default=0),
            max([val for val in metric_data['mae'] if not np.isnan(val)], default=0)
        ])
        plt.ylim(0, max_error * 1.2)  # Add 20% margin
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Create summary table as DataFrame
        summary_data = []
        for i, model_name in enumerate(model_names):
            row = {'Model': model_name, 'Type': model_types[i]}
            for metric in metrics:
                value = metric_data[metric][i]
                if not np.isnan(value):
                    row[metric] = value
                else:
                    row[metric] = 'N/A'  # Use N/A for missing metrics in the table
            
            # Add additional metrics if available
            for extra_metric in ['precision', 'recall', 'r2']:
                if extra_metric in valid_results[model_name] and valid_results[model_name][extra_metric] is not None:
                    row[extra_metric] = valid_results[model_name][extra_metric]
                else:
                    row[extra_metric] = 'N/A'
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Print summary table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print("\nModel Comparison Summary:")
        print(summary_df)
        
        # Save to CSV
        summary_df.to_csv('model_comparison_results.csv', index=False)
        
        return summary_df
    
    def analyze_error_distribution(self, results, X_test, y_test):
        """Analyze error distribution by true rating value."""
        if not results:
            print("No results to analyze.")
            return
        
        # Filter out models with errors
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid models to analyze error distribution.")
            return
        
        # Create dataframe for error analysis
        y_test_discrete = np.round(y_test).astype(int)
        error_df = pd.DataFrame({'true_rating': y_test_discrete})
        
        # Add prediction and error columns for each model
        for model_name, result in valid_results.items():
            if 'predictions' in result:
                error_df[f'{model_name}_pred'] = result['predictions']
                error_df[f'{model_name}_error'] = np.abs(error_df[f'{model_name}_pred'] - error_df['true_rating'])
        
        # Group by true rating
        group_by_rating = error_df.groupby('true_rating')
        
        # Calculate mean absolute error by true rating for each model
        mae_by_rating = pd.DataFrame()
        for model_name in valid_results.keys():
            if f'{model_name}_error' in error_df.columns:
                mae_by_rating[model_name] = group_by_rating[f'{model_name}_error'].mean()
        
        # Plot MAE by true rating
        if not mae_by_rating.empty:
            plt.figure(figsize=(12, 8))
            for model_name in mae_by_rating.columns:
                plt.plot(mae_by_rating.index, mae_by_rating[model_name], marker='o', label=model_name)
            
            plt.xlabel('True Rating')
            plt.ylabel('Mean Absolute Error')
            plt.title('Error Distribution by True Rating')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('error_distribution.png')
            print("Error distribution plot saved as 'error_distribution.png'")
        else:
            print("No error data to plot.")
        
        # Calculate distribution of true ratings
        rating_counts = group_by_rating.size()
        rating_percent = rating_counts / rating_counts.sum() * 100
        
        # Print rating distribution
        print("\nRating Distribution in Test Set:")
        for rating, count in rating_counts.items():
            print(f"Rating {rating}: {count} samples ({rating_percent[rating]:.2f}%)")
        
        # Create a heatmap of confusion matrices for classifier models
        for model_name, result in valid_results.items():
            if result.get('model_type') == 'classifier' and 'predictions' in result:
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test_discrete, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=sorted(np.unique(y_test_discrete)),
                           yticklabels=sorted(np.unique(y_test_discrete)))
                plt.xlabel('Predicted Rating')
                plt.ylabel('True Rating')
                plt.title(f'Confusion Matrix for {model_name}')
                plt.tight_layout()
                plt.savefig(f'{model_name}_confusion_matrix.png')
                print(f"Confusion matrix saved as '{model_name}_confusion_matrix.png'")
        
        return mae_by_rating

def main():
    """Main function to evaluate and compare all models."""
    print("Starting comprehensive model evaluation...")
    
    # Create evaluator instance
    evaluator = ModelEvaluator()
    
    try:
        # Load test data
        test_data = evaluator.load_data()
        
        # Prepare features
        X_test, y_test = evaluator.prepare_features(test_data)
        
        # Load all models
        evaluator.load_models()
        
        # Evaluate all models
        results = evaluator.evaluate_all_models(X_test, y_test)
        
        # Plot comparison
        summary = evaluator.plot_comparison(results)
        
        # Analyze error distribution
        evaluator.analyze_error_distribution(results, X_test, y_test)
        
        print("\nModel comparison completed successfully!")
        
        # Print the best model for each metric
        if not summary.empty:
            print("\nBest model by metric:")
            metrics = ['accuracy', 'within_one', 'rmse', 'mae', 'f1']
            
            for metric in metrics:
                if metric in summary.columns and not summary[metric].isna().all():
                    if metric in ['rmse', 'mae']:  # Lower is better
                        valid_values = summary[~summary[metric].isna()][metric]
                        if not valid_values.empty:
                            best_model = valid_values.idxmin()
                            best_value = valid_values.min()
                            print(f"Best by {metric.upper()}: {best_model} ({best_value:.4f})")
                    else:  # Higher is better
                        valid_values = summary[~summary[metric].isna()][metric]
                        if not valid_values.empty:
                            best_model = valid_values.idxmax()
                            best_value = valid_values.max()
                            print(f"Best by {metric.upper()}: {best_model} ({best_value:.4f})")
        
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Model comparison failed to complete.")

if __name__ == "__main__":
    main() 