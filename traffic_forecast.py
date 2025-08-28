import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# If you need to install the holidays package, run:
# pip install holidays

import holidays

# --- 1. Load Data ---
# Load the training and test datasets
dset1 = 'Smart-Traffic-Prediction-main/datasets/train_aWnotuB.csv'
dset2 = 'Smart-Traffic-Prediction-main/datasets/datasets_8494_11879_test_BdBKkAj.csv'
train_df = pd.read_csv(dset1)
test_df = pd.read_csv(dset2)

# --- 2. Feature Engineering ---
# Convert 'DateTime' column to datetime objects
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'], format='%d-%m-%Y %H:%M')
test_df['DateTime'] = pd.to_datetime(test_df['DateTime'], format='%d-%m-%Y %H:%M')

# Create a feature for Indian holidays
india_holidays = holidays.India()
for df in [train_df, test_df]:
    df['Is_Holiday'] = df['DateTime'].dt.date.astype('datetime64[ns]').isin(india_holidays)

# Extract time-based features from the 'DateTime' column
for df in [train_df, test_df]:
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['Weekday'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6

# --- 3. Data Preparation ---
# Convert the 'Junction' column to a numerical format
le = LabelEncoder()
train_df['Junction'] = le.fit_transform(train_df['Junction'].astype(str))
test_df['Junction'] = le.transform(test_df['Junction'].astype(str))

# --- 4. Model Training ---
# Define the features (X) and the target (y)
features = ['Year', 'Month', 'Day', 'Hour', 'Weekday', 'Junction', 'Is_Holiday']
target = 'Vehicles'

X = train_df[features]
y = train_df[target]

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the LightGBM model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Initialize and train the LightGBM model
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=True)])

# --- 5. Prediction ---
# Make predictions on the test data
test_predictions = model.predict(test_df[features])

# --- 6. Create Submission File ---
# Create a new DataFrame for the submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Vehicles': test_predictions})

# Convert the predictions to integers
submission_df['Vehicles'] = submission_df['Vehicles'].astype(int)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission_df.head())