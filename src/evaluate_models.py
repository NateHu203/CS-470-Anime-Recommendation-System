import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, 
    mean_absolute_error, 
    mean_squared_error, 
    f1_score, 
    recall_score, 
    roc_auc_score, 
    r2_score
)

print("Loading test data and applying the same preprocessing as in version1.py...")

# Read data (using the same paths as version1.py)
rating_test = pd.read_csv('../data/processed/rating_test_v2_new.csv')
anime_feature = pd.read_csv('../data/processed/anime_features_normalized_new.csv')

# Drop the Unnamed column from anime_feature if it exists (matching version1.py)
if 'Unnamed: 10' in anime_feature.columns:
    anime_feature = anime_feature.drop('Unnamed: 10', axis=1)

# Merge data (exactly as in version1.py)
merged_test = rating_test.merge(anime_feature, on='anime_id', how='left')

# Identify columns with '_x' suffix from the merge - these are duplicate columns that need special handling
x_columns = [col for col in merged_test.columns if col.endswith('_x')]
columns_to_normalize = []

for col in x_columns:
    base_col = col[:-2]  # Remove _x suffix
    # Add to normalization list if it matches our criteria
    if base_col in ['user_median_rating', 'user_rating_std']:
        columns_to_normalize.append(col)

# Add other columns that need normalization
additional_norm_columns = ['user_rating_count', 'engagement_percentile', 'genre_diversity', 'user_mean_rating']
for col in additional_norm_columns:
    if col in merged_test.columns:
        columns_to_normalize.append(col)

# Ensure all normalization columns exist
columns_to_normalize = [col for col in columns_to_normalize if col in merged_test.columns]

# Data preprocessing - normalize specified columns
if columns_to_normalize:
    # Handle NaN values before scaling
    merged_test[columns_to_normalize] = merged_test[columns_to_normalize].fillna(0)
    
    # Note: We should use the same scaler as training but we don't have it
    # This is a limitation, but we'll use a new scaler as best effort
    scaler = MinMaxScaler()
    merged_test[columns_to_normalize] = scaler.fit_transform(merged_test[columns_to_normalize])

# Process top_genres column if it exists
if 'top_genres' in merged_test.columns:
    # Convert string representations of lists to actual lists
    merged_test['top_genres'] = merged_test['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
    
    # Ideally, we should use the same MLB from training, but we'll create one
    mlb = MultiLabelBinarizer()
    mlb.fit(merged_test['top_genres'])  # This isn't ideal, but we don't have the trained MLB
    genre_test_matrix = mlb.transform(merged_test['top_genres'])
    genre_test_df = pd.DataFrame(genre_test_matrix, columns=mlb.classes_)
    
    merged_test = pd.concat([merged_test.reset_index(drop=True), genre_test_df], axis=1)

# Drop unnecessary columns
columns_to_drop = ['user_id', 'anime_id']

# Add rating_y to columns_to_drop if it exists (this is the duplicate rating column from merge)
if 'rating_y' in merged_test.columns:
    columns_to_drop.append('rating_y')

# Drop columns
merged_test = merged_test.drop(columns_to_drop, axis=1)

# Identify target variable column
target_column = 'rating'
if target_column not in merged_test.columns and 'rating_x' in merged_test.columns:
    target_column = 'rating_x'

# Separate features and target variables
columns_to_exclude = [target_column, 'top_genres', 'name']
X_test = merged_test.drop([col for col in columns_to_exclude if col in merged_test.columns], axis=1)
y_test = merged_test[target_column]

# Remove rows with NaN values
X_test = X_test.dropna()
y_test = y_test[X_test.index]

print(f"Test data processed. Number of features: {len(X_test.columns)}")
print(f"Number of test samples: {len(X_test)}")

# Function to calculate "within 1 point accuracy"
def within_one_accuracy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 1)

# Function to evaluate a model with all metrics (similar to version1.py's evaluate_model function)
def evaluate_model(model, X_test, y_test, model_name):
    # Check and align feature columns if needed
    if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
        model_features = model.best_estimator_.feature_names_in_
        # Check if test features match model features
        missing_features = [f for f in model_features if f not in X_test.columns]
        extra_features = [f for f in X_test.columns if f not in model_features]
        if missing_features:
            print(f"Warning: {len(missing_features)} features used by the model are missing in test data")
            print(f"Missing features: {missing_features[:5]}..." if len(missing_features) > 5 else f"Missing features: {missing_features}")
        if extra_features:
            print(f"Warning: {len(extra_features)} features in test data not used by the model")
            print(f"Extra features: {extra_features[:5]}..." if len(extra_features) > 5 else f"Extra features: {extra_features}")
        
        # Align columns - use only features that exist in both
        common_features = [f for f in model_features if f in X_test.columns]
        if common_features:
            print(f"Using {len(common_features)} common features")
            X_test_aligned = X_test[common_features]
        else:
            print("No common features found. Cannot evaluate model.")
            return None
    else:
        X_test_aligned = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_aligned)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    within_1 = within_one_accuracy(y_test, y_pred)
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_aligned), multi_class='ovr')
        auc_str = f"AUC: {auc:.4f}"
    except (ValueError, AttributeError):
        auc = None
        auc_str = "AUC: Unable to calculate"
    
    # Print best parameters if available
    if hasattr(model, 'best_params_'):
        print(f"{model_name} - Best Parameters: {model.best_params_}")
    
    if hasattr(model, 'best_score_'):
        print(f"Cross-Validation Score: {model.best_score_:.4f}")
    
    # Print results
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Within 1 Point Accuracy: {within_1:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"{auc_str}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'within_1_accuracy': within_1,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'f1': f1,
        'recall': recall,
        'auc': auc
    }

# Load models
models_dir = '../src/models'
results = []

# Check if models exist
dt_path = os.path.join(models_dir, 'decision_tree_classifier.pkl')
rf_path = os.path.join(models_dir, 'random_forest_classifier.pkl')
ada_path = os.path.join(models_dir, 'adaboost_classifier.pkl')

print("\nLoading and evaluating models...")

if os.path.exists(dt_path):
    print("Loading Decision Tree model...")
    dt_model = joblib.load(dt_path)
    dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    if dt_result:
        results.append(dt_result)
else:
    print(f"Model not found: {dt_path}")

if os.path.exists(rf_path):
    print("Loading Random Forest model...")
    rf_model = joblib.load(rf_path)
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    if rf_result:
        results.append(rf_result)
else:
    print(f"Model not found: {rf_path}")

if os.path.exists(ada_path):
    print("Loading AdaBoost model...")
    ada_model = joblib.load(ada_path)
    ada_result = evaluate_model(ada_model, X_test, y_test, "AdaBoost")
    if ada_result:
        results.append(ada_result)
else:
    print(f"Model not found: {ada_path}")

# Compare models
if results:
    print("\n=== Model Comparison ===")
    print("Model\tAccuracy\tWithin 1\tMAE\tRMSE\tR²\tF1 Score")
    for result in results:
        print(f"{result['model']}\t{result['accuracy']:.4f}\t{result['within_1_accuracy']:.4f}\t{result['mae']:.4f}\t{result['rmse']:.4f}\t{result['r2']:.4f}\t{result['f1']:.4f}")

    # Print best overall model by different metrics
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_within1 = max(results, key=lambda x: x['within_1_accuracy'])
    best_mae = min(results, key=lambda x: x['mae'])
    best_rmse = min(results, key=lambda x: x['rmse'])
    best_r2 = max(results, key=lambda x: x['r2'])
    
    print("\n=== Best Models by Metric ===")
    print(f"Best by Accuracy: {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"Best by Within 1 Point Accuracy: {best_within1['model']} ({best_within1['within_1_accuracy']:.4f})")
    print(f"Best by MAE: {best_mae['model']} ({best_mae['mae']:.4f})")
    print(f"Best by RMSE: {best_rmse['model']} ({best_rmse['rmse']:.4f})")
    print(f"Best by R²: {best_r2['model']} ({best_r2['r2']:.4f})")
else:
    print("No models were successfully evaluated.") 