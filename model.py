from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_classifier(X_train, y_train):
    # Step 2: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 3: Impute missing values if any
    imputer = SimpleImputer(strategy='mean')  # You can use other strategies as well
    X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)

    # Step 4: Train the KNN model
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors as needed
    knn_classifier.fit(X_train_scaled_imputed, y_train)

    # Step 5: Make predictions on the training data
    y_pred = knn_classifier.predict(X_train_scaled_imputed)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_train, y_pred)
    print(f'Accuracy: {accuracy}')

    # Note: In a real-world scenario, you would ideally evaluate on a separate testing set.
    # Using the training set for evaluation can give a misleadingly optimistic view of performance.
    # Save the model and use it for predictions on new, unseen data.
    return knn_classifier
