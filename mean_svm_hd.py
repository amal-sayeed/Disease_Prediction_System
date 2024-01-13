import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=20)

# Initialize and fit an SVM model
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train, y_train)

# Predict the target values on the test data using SVM
y_pred_svm = svm_model.predict(X_test)

# Calculate the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Calculate precision and recall
precision_svm = precision_score(y_test, y_pred_svm, pos_label=1)  # Assuming 4 represents the malignant class
recall_svm = recall_score(y_test, y_pred_svm, pos_label=1)

print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)

# Calculate and print the confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:")
print(conf_matrix_svm)

