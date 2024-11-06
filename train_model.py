import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle

# Load data
data = pd.read_csv('Hyderbad_House_price.csv')

# Check if 'location' column is present
if 'location' not in data.columns:
    raise ValueError("The 'location' column is missing in the CSV file. Please include the 'location' column for prediction.")

# Drop unnecessary column if present
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Extract BHK feature from 'title' column
def extract_bhk(title):
    match = re.search(r'(\d+) BHK', title)
    return int(match.group(1)) if match else 0

data['bhk'] = data['title'].apply(extract_bhk)
data = data.drop(columns=['title'])  # Drop title column as it's no longer needed

# Handling missing values (if any)
data.fillna(data.mean(), inplace=True)  # Impute missing numerical values with column mean

# Define features and target variable
X = data.drop(columns=['price(L)'])
y = np.log1p(data['price(L)'])  # Log-transform the target variable

# Define categorical and numerical features
categorical_features = ['location', 'building_status']
numerical_features = ['rate_persqft', 'area_insqft', 'bhk']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features),  # Scaling numerical features
    ],
    remainder='passthrough'
)

# Create pipeline with RandomForestRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Inverse the log transformation to get actual house price
y_pred_actual = np.expm1(y_pred)  # expm1 is the inverse of log1p

# Evaluate the model
mse = mean_squared_error(np.expm1(y_test), y_pred_actual)  # Inverse log transformation for actual test data
r2 = r2_score(np.expm1(y_test), y_pred_actual)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Perform K-fold Cross Validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)

# Save the model with the best hyperparameters
with open('model.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)
