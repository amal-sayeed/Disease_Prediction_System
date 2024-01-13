import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Step 1: Load the dataset without column names
data = pd.read_csv('combined heart disease.csv')

# Step 2: Replace '?' with NaN
data.replace('?', float('nan'), inplace=True)

# Step 3: Create a KNN imputer
knn_imputer = KNNImputer(n_neighbors=5)  # Choose an appropriate value for 'n_neighbors'

# Impute missing values
data_imputed = knn_imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed)

# Step 4: Designate the last column as the target column
X = data_imputed.iloc[:, :-1]  # Features (all columns except the last one)
y = data_imputed.iloc[:, -1]   # Target (the last column)

# Step 5: Split the data into training and testing sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Create a logistic regression model with increased max_iter
logistic_model = LogisticRegression(max_iter=1000)

# Train the model on the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with KNN imputation and Logistic Regression: {accuracy:.4f}")

# Calculate precision and recall
precision_logistic = precision_score(y_test, y_pred, pos_label=1, zero_division=1)  # Assuming 4 represents the malignant class
recall_logistic = recall_score(y_test, y_pred, pos_label=1)

print("Logistic Regression Precision:", precision_logistic)
print("Logistic Regression Recall:", recall_logistic)

# Calculate and print the confusion matrix
conf_matrix_logistic = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix_logistic)
