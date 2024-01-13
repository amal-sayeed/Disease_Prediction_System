import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset without column names
data = pd.read_csv('wisconsin_breast_cancer_actual-2.csv', header=None)

# Step 2: Replace '?' with NaN
data.replace('?', float('nan'), inplace=True)

# Step 3: Designate the last column as the target column
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target (the last column)

# Step 4: Split the data into training and testing sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Iterate over different values of n_neighbors
for n_neighbors in range(1, 20):  # You can adjust the range of n_neighbors as needed
    # Create a KNN imputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    # Impute missing values separately for training and testing sets
    X_train_imputed = knn_imputer.fit_transform(X_train)
    X_test_imputed = knn_imputer.transform(X_test)

    # Create an SVM classifier
    svm_model = SVC()

    # Train the model on the training data
    svm_model.fit(X_train_imputed, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test_imputed)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with KNN imputation (n_neighbors={n_neighbors}) and SVM: {accuracy:.4f}")
