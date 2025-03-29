import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load the CSV file
file_path = "C:\\Users\\chris\\Documents\\Data Science Course\\Course 1\\Mini Project 5.3\\Raw Data.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Step 1: Check for missing values
missing_values = df.isnull().sum()

# Step 2: Check for duplicates
duplicate_rows = df.duplicated().sum()

# Step 3: Standardize column names (lowercase, replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(" ", "_", regex=True)

# Step 4: Check data types
data_types = df.dtypes

# Step 5: Detect outliers using IQR method and filters for only numeric columns
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
numeric_cols = df.select_dtypes(include=[np.number])
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1
outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).sum()

# Summarize preprocessing checks
preprocessing_summary = {
    "Missing Values": missing_values,
    "Duplicate Rows": duplicate_rows,
    "Data Types": data_types,
    "Outliers Detected": outliers
}
print(preprocessing_summary)

# Generate descriptive statistics
descriptive_stats = df.describe()

# Calculate median for each feature
median_values = df.median()

# Identify 95th percentile values for two features
features_to_check = ["engine_rpm", "fuel_pressure"]
percentile_95 = df[features_to_check].quantile(0.95)

# Identify range values beyond the 95th percentile
outlier_ranges = {}
for feature in features_to_check:
    threshold = percentile_95[feature]
    outlier_ranges[feature] = df[df[feature] > threshold][feature]

# Compile results
stats_summary = {
    "Descriptive Statistics": descriptive_stats,
    "Median Values": median_values,
    "95th Percentile Values": percentile_95,
    "Values Beyond 95th Percentile": outlier_ranges,
}

from pprint import pprint
pprint(stats_summary)

# Create subplots for histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram for Engine RPM
sns.histplot(df["engine_rpm"], bins=50, kde=True, ax=axes[0], color="blue")
axes[0].axvline(df["engine_rpm"].quantile(0.95), color='red', linestyle='dashed', label="95th Percentile")
axes[0].set_title("Engine RPM Distribution")
axes[0].set_xlabel("Engine RPM")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Histogram for Fuel Pressure
sns.histplot(df["fuel_pressure"], bins=50, kde=True, ax=axes[1], color="green")
axes[1].axvline(df["fuel_pressure"].quantile(0.95), color='red', linestyle='dashed', label="95th Percentile")
axes[1].set_title("Fuel Pressure Distribution")
axes[1].set_xlabel("Fuel Pressure")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# Show the plots
plt.tight_layout()
plt.show()

# Create subplots for histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram for Engine RPM
sns.histplot(df["engine_rpm"], bins=50, kde=True, ax=axes[0], color="blue")
axes[0].axvline(df["engine_rpm"].quantile(0.95), color='red', linestyle='dashed', label="95th Percentile")
axes[0].set_title("Engine RPM Distribution")
axes[0].set_xlabel("Engine RPM")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Histogram for Fuel Pressure
sns.histplot(df["fuel_pressure"], bins=50, kde=True, ax=axes[1], color="green")
axes[1].axvline(df["fuel_pressure"].quantile(0.95), color='red', linestyle='dashed', label="95th Percentile")
axes[1].set_title("Fuel Pressure Distribution")
axes[1].set_xlabel("Fuel Pressure")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# Show the plots
plt.tight_layout()
plt.show()

# Define a function to identify outliers based on IQR
def identify_outliers_iqr(df, features):
    outliers = pd.DataFrame(index=df.index)
    
    for feature in features:
        # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR for each feature
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers: 1 if outlier, 0 if not
        outliers[feature] = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).astype(int)
    
    return outliers

# List of features to check for outliers
features_to_check = ["engine_rpm", "fuel_pressure"]  # Adjust this list as per your dataset

# Identify outliers for each feature
outliers = identify_outliers_iqr(df, features_to_check)

# Create a new column 'outlier_flag' that indicates if two or more features have outliers for a sample
outliers['outlier_flag'] = outliers.sum(axis=1) >= 2

# Add the outlier information to the original DataFrame
df['outlier_flag'] = outliers['outlier_flag']

# Display the updated DataFrame with the new outlier flag
print(df[['engine_rpm', 'fuel_pressure', 'outlier_flag']]) 

from sklearn.preprocessing import StandardScaler

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features_to_check])  

# scale only selected features
from sklearn.svm import OneClassSVM

# Define the One-Class SVM model (with default parameters first)
svm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)  # nu defines the fraction of outliers
svm_model.fit(df_scaled)

# Predict anomalies (1 for inliers, -1 for outliers)
svm_predictions = svm_model.predict(df_scaled)
svm_predictions = [0 if x == 1 else 1 for x in svm_predictions]  # Convert to binary (0, 1)
df['svm_outlier'] = svm_predictions

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA to reduce to 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualize the results
plt.figure(figsize=(8,6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['svm_outlier'], cmap='coolwarm', edgecolors='k')
plt.title("One-Class SVM Anomaly Detection (PCA Visualization)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Hyperparameter tuning with cross-validation
svm_model_tuned = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.02)
svm_model_tuned.fit(df_scaled)
svm_predictions_tuned = svm_model_tuned.predict(df_scaled)
svm_predictions_tuned = [0 if x == 1 else 1 for x in svm_predictions_tuned]
df['svm_outlier_tuned'] = svm_predictions_tuned

from sklearn.ensemble import IsolationForest

# Define the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)  # contamination is expected fraction of outliers
iso_forest.fit(df_scaled)

# Predict anomalies (1 for inliers, -1 for outliers)
iso_predictions = iso_forest.predict(df_scaled)
iso_predictions = [0 if x == 1 else 1 for x in iso_predictions]  # Convert to binary (0, 1)
df['iso_outlier'] = iso_predictions

# Perform PCA to reduce to 2D
df_pca_iso = pca.fit_transform(df_scaled)

# Visualize the results
plt.figure(figsize=(8,6))
plt.scatter(df_pca_iso[:, 0], df_pca_iso[:, 1], c=df['iso_outlier'], cmap='coolwarm', edgecolors='k')
plt.title("Isolation Forest Anomaly Detection (PCA Visualization)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Hyperparameter tuning with Isolation Forest
iso_forest_tuned = IsolationForest(n_estimators=200, contamination=0.03, max_samples='auto', random_state=42)
iso_forest_tuned.fit(df_scaled)
iso_predictions_tuned = iso_forest_tuned.predict(df_scaled)
iso_predictions_tuned = [0 if x == 1 else 1 for x in iso_predictions_tuned]
df['iso_outlier_tuned'] = iso_predictions_tuned
