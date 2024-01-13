import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load the dataset with missing values denoted by '?'
data = pd.read_csv("combined heart disease.csv")

# Replace '?' with NaN
data.replace('?', float('nan'), inplace=True)

# Extract the feature columns (X) and target column (y)
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Create an imputer with "mean" strategy
imputer = SimpleImputer(strategy="mean")

# Impute missing values with mean
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and fit a Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Initialize and fit an AdaBoost model with Logistic Regression as the base estimator
adaboost_model = AdaBoostClassifier(estimator=logistic_model, n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)

# Predict the target values on the test data using AdaBoost
y_pred_adaboost = adaboost_model.predict(X_test)

# Calculate the accuracy of the AdaBoost model
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print("AdaBoost Accuracy:", accuracy_adaboost)

# Calculate precision and recall with explicit pos_label
precision_adaboost = precision_score(y_test, y_pred_adaboost, pos_label=1)
recall_adaboost = recall_score(y_test, y_pred_adaboost, pos_label=1)

print("AdaBoost Precision:", precision_adaboost)
print("AdaBoost Recall:", recall_adaboost)

# Calculate and print the confusion matrix
conf_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)
print("Confusion Matrix:")
print(conf_matrix_adaboost)

