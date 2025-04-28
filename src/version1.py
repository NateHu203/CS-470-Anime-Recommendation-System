import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, recall_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import time
import os
import joblib
from sklearn.model_selection import train_test_split

# Create models directory if it doesn't exist
os.makedirs('../src/models', exist_ok=True)

# Read data
rating_train = pd.read_csv('../data/processed/rating_train_v2_new.csv')
rating_test = pd.read_csv('../data/processed/rating_test_v2_new.csv')
anime_feature = pd.read_csv('../data/processed/anime_features_normalized_new.csv')

# Drop the Unnamed column from anime_feature if it exists
if 'Unnamed: 10' in anime_feature.columns:
    anime_feature = anime_feature.drop('Unnamed: 10', axis=1)

# Merge data
merged_data = rating_train.merge(anime_feature, on='anime_id', how='left')
merged_test = rating_test.merge(anime_feature, on='anime_id', how='left')

# Identify columns with '_x' suffix from the merge - these are duplicate columns that need special handling
x_columns = [col for col in merged_data.columns if col.endswith('_x')]
columns_to_normalize = []

for col in x_columns:
    base_col = col[:-2]  # Remove _x suffix
    # Add to normalization list if it matches our criteria
    if base_col in ['user_median_rating', 'user_rating_std']:
        columns_to_normalize.append(col)

# Add other columns that need normalization
additional_norm_columns = ['user_rating_count', 'engagement_percentile', 'genre_diversity', 'user_mean_rating']
for col in additional_norm_columns:
    if col in merged_data.columns:
        columns_to_normalize.append(col)

# Ensure all normalization columns exist
columns_to_normalize = [col for col in columns_to_normalize if col in merged_data.columns]

# Data preprocessing - normalize specified columns
if columns_to_normalize:
    # Handle NaN values before scaling
    merged_data[columns_to_normalize] = merged_data[columns_to_normalize].fillna(0)
    merged_test[columns_to_normalize] = merged_test[columns_to_normalize].fillna(0)
    
    scaler = MinMaxScaler()
    merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])
    merged_test[columns_to_normalize] = scaler.transform(merged_test[columns_to_normalize])

# Process top_genres column if it exists
if 'top_genres' in merged_data.columns:
    # Convert string representations of lists to actual lists
    merged_data['top_genres'] = merged_data['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
    merged_test['top_genres'] = merged_test['top_genres'].str.strip('[]').str.replace("'", "").str.split(', ')
    
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(merged_data['top_genres'])
    genre_test_matrix = mlb.transform(merged_test['top_genres'])
    
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
    genre_test_df = pd.DataFrame(genre_test_matrix, columns=mlb.classes_)
    
    merged_data = pd.concat([merged_data.reset_index(drop=True), genre_df], axis=1)
    merged_test = pd.concat([merged_test.reset_index(drop=True), genre_test_df], axis=1)

# Drop unnecessary columns
columns_to_drop = ['user_id', 'anime_id']

# Add rating_y to columns_to_drop if it exists (this is the duplicate rating column from merge)
if 'rating_y' in merged_data.columns:
    columns_to_drop.append('rating_y')

# Drop columns
merged_data = merged_data.drop(columns_to_drop, axis=1)
merged_test = merged_test.drop(columns_to_drop, axis=1)

# Identify target variable column
target_column = 'rating'
if target_column not in merged_data.columns and 'rating_x' in merged_data.columns:
    target_column = 'rating_x'

# Separate features and target variables
columns_to_exclude = [target_column, 'top_genres', 'name']
X_train = merged_data.drop([col for col in columns_to_exclude if col in merged_data.columns], axis=1)
X_test = merged_test.drop([col for col in columns_to_exclude if col in merged_test.columns], axis=1)
y_train = merged_data[target_column]
y_test = merged_test[target_column]

print(f"Feature columns: {X_train.columns.tolist()}")
print(f"Number of features: {len(X_train.columns)}")

# Remove rows with NaN values
X_train = X_train.dropna()
y_train = y_train[X_train.index]
X_test = X_test.dropna()
y_test = y_test[X_test.index]

# Take a subset of 10,000 samples for GridSearchCV to save space and time
if len(X_train) > 10000:
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, train_size=10000, random_state=42, stratify=y_train
    )
    print(f"Using 10,000 samples out of {len(X_train)} for GridSearchCV")
else:
    X_train_sample = X_train
    y_train_sample = y_train
    print(f"Using all {len(X_train)} samples for GridSearchCV")

# Define evaluation function for model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        auc_str = f"AUC: {auc:.4f}"
    except ValueError:
        auc_str = "AUC: Unable to calculate"
        
    print(f"{model_name} - Best Parameters: {model.best_params_}")
    print(f"Cross-Validation Score: {model.best_score_:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"{auc_str}")
    print(f"RMSE: {rmse:.4f}")
    print()
    
    return {
        'model': model_name,
        'best_params': model.best_params_,
        'cv_score': model.best_score_,
        'accuracy': accuracy,
        'mae': mae,
        'f1': f1,
        'recall': recall,
        'rmse': rmse
    }

# Function to save model to the models directory
def save_model(model, filename):
    filepath = f'../src/models/{filename}'
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
    # Save feature importance if the model supports it
    if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_filepath = '../src/models/feature_importance.csv'
        feature_importances.to_csv(importance_filepath, index=False)
        print(f"Feature importance saved to {importance_filepath}")

# Hyperparameter grids for different models
dt_param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}

adaboost_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1.0]
}

# List to store results
all_results = []

# Decision Tree with GridSearchCV
print("Training Decision Tree model with GridSearchCV...")
start_time = time.time()
dt_grid = GridSearchCV(
    DecisionTreeClassifier(),
    dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
dt_grid.fit(X_train_sample, y_train_sample)
dt_time = time.time() - start_time
print(f"Decision Tree training completed in {dt_time:.2f} seconds\n")

# Train the final model on the entire dataset using the best parameters
final_dt = DecisionTreeClassifier(**dt_grid.best_params_)
final_dt.fit(X_train, y_train)

# Save the model
save_model(dt_grid, 'decision_tree_classifier.pkl')

dt_results = evaluate_model(dt_grid, X_test, y_test, "Decision Tree")
dt_results['training_time'] = dt_time
all_results.append(dt_results)

# Random Forest with GridSearchCV
print("Training Random Forest model with GridSearchCV...")
start_time = time.time()
rf_grid = GridSearchCV(
    RandomForestClassifier(),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_sample, y_train_sample)
rf_time = time.time() - start_time
print(f"Random Forest training completed in {rf_time:.2f} seconds\n")

# Train the final model on the entire dataset using the best parameters
final_rf = RandomForestClassifier(**rf_grid.best_params_)
final_rf.fit(X_train, y_train)

# Save the model
save_model(rf_grid, 'random_forest_classifier.pkl')

rf_results = evaluate_model(rf_grid, X_test, y_test, "Random Forest")
rf_results['training_time'] = rf_time
all_results.append(rf_results)

# AdaBoost with GridSearchCV
print("Training AdaBoost model with GridSearchCV...")
start_time = time.time()
ada_grid = GridSearchCV(
    AdaBoostClassifier(DecisionTreeClassifier()),
    adaboost_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
ada_grid.fit(X_train_sample, y_train_sample)
ada_time = time.time() - start_time
print(f"AdaBoost training completed in {ada_time:.2f} seconds\n")

# Train the final model on the entire dataset using the best parameters
final_ada = AdaBoostClassifier(DecisionTreeClassifier(), **ada_grid.best_params_)
final_ada.fit(X_train, y_train)

# Save the model
save_model(ada_grid, 'adaboost_classifier.pkl')

ada_results = evaluate_model(ada_grid, X_test, y_test, "AdaBoost")
ada_results['training_time'] = ada_time
all_results.append(ada_results)

# Compare models
print("\n=== Model Comparison ===")
print("Model\tAccuracy\tF1 Score\tTraining Time (s)")
for result in all_results:
    print(f"{result['model']}\t{result['accuracy']:.4f}\t{result['f1']:.4f}\t{result['training_time']:.2f}")

# Print best overall model
best_model = max(all_results, key=lambda x: x['accuracy'])
print(f"\nBest model: {best_model['model']} with accuracy: {best_model['accuracy']:.4f}")
print(f"Best parameters: {best_model['best_params']}")