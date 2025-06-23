"""
Credit Card Fraud Detection using Unsupervised Deep Learning (AutoEncoder)

This Python script uses the PyOD library's AutoEncoder model to identify fraudulent credit card transactions from the Kaggle dataset.
The process includes loading data, normalizing values, training an unsupervised model, predicting anomalies, and evaluating performance.
"""

# -------------------------------
# Step 1: Load Dataset using Pandas
# -------------------------------
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("creditcard.csv")

# Display the dataset's shape, column names, and first few rows for inspection
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -------------------------------
# Step 2: Normalize 'Time' and 'Amount'
# -------------------------------
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler for feature normalization
scaler = StandardScaler()

# Apply standard scaling to 'Time' and 'Amount' columns
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Separate features (X) and target labels (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Print shape of feature matrix and distribution of class labels
print("Feature matrix shape:", X.shape)
print("Labels distribution:\n", y.value_counts())

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes and fraud counts in each set to confirm class distribution
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training fraud cases:", y_train.sum())
print("Testing fraud cases:", y_test.sum())

# -------------------------------
# Step 4: Initialize and Train AutoEncoder
# -------------------------------
from pyod.models.auto_encoder import AutoEncoder

# Configure the AutoEncoder with a symmetrical neural network architecture
autoencoder = AutoEncoder(
    hidden_neuron_list=[64, 32, 32, 64],
    contamination=0.001
)

# Fit the AutoEncoder model on training data (unsupervised learning)
autoencoder.fit(X_train)

# Predict anomalies on test set (1 = fraud/anomaly, 0 = normal)
y_test_pred = autoencoder.predict(X_test)

# Get anomaly scores used for AUC evaluation
y_test_scores = autoencoder.decision_function(X_test)

print("Anomaly prediction complete.")

# -------------------------------
# Step 5: Evaluate Model Performance
# -------------------------------
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Print confusion matrix to see counts of TP, FP, TN, FN
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Print precision, recall, and F1-score for both classes
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

# Compute ROC-AUC score based on prediction scores and true labels
auc_score = roc_auc_score(y_test, y_test_scores)

# Print AUC score which indicates model's discrimination ability
print(f"\nROC-AUC Score: {auc_score:.4f}")
