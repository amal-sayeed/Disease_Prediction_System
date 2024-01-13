import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset without column names
data = pd.read_csv('wisconsin_breast_cancer_actual-2.csv')

# Step 2: Replace '?' with NaN
data.replace('?', float('nan'), inplace=True)

# Step 3: Designate the last column as the target column
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target (the last column)

# Step 4: Split the data into training and testing sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Iterate over different values of k
for k in range(1, 20):  # You can adjust the range of k as needed
    # Create a KNN imputer
    knn_imputer = KNNImputer(n_neighbors=k)

    # Impute missing values separately for training and testing sets
    X_train_imputed = knn_imputer.fit_transform(X_train)
    X_test_imputed = knn_imputer.transform(X_test)

    # Create a logistic regression model
    logistic_model = LogisticRegression()

    # Train the model on the training data
    logistic_model.fit(X_train_imputed, y_train)

    # Make predictions on the test data
    y_pred = logistic_model.predict(X_test_imputed)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with KNN imputation (k={k}) and Logistic Regression: {accuracy:.4f}")

