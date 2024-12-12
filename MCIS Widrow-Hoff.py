import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import os

# Fix KMeans memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize Data (Prevent Invalid Weight Updates)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term for Widrow-Hoff
X = np.hstack([X, np.ones((X.shape[0], 1))])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MCIS Algorithm
def mcis(X, y, k=3, r=1.5):
    """Multi-Class Instance Selection (MCIS)"""
    selected_indices = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        X_pos = X[y == cls]
        X_neg = X[y != cls]
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pos)
        centers = kmeans.cluster_centers_
        for center in centers:
            distances = np.linalg.norm(X_neg - center, axis=1)
            selected_indices.extend(np.where(distances < r)[0])
    selected_indices = np.unique(selected_indices).astype(int)
    if len(selected_indices) == 0:
        print("MCIS filtering resulted in no samples. Using all data as fallback.")
        selected_indices = np.arange(len(X))
    return selected_indices

# Apply MCIS
selected_indices = mcis(X_train, y_train, k=3, r=3.0)  # Adjust k and r as needed
X_train_filtered = X_train[selected_indices]
y_train_filtered = y_train[selected_indices]

# Debug: Print filtered data shape
print(f"Filtered Training Data Shape: {X_train_filtered.shape}")
print(f"Filtered Labels Shape: {y_train_filtered.shape}")

# Widrow-Hoff Algorithm
class WidrowHoff:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        for epoch in range(self.epochs):  # Fixed loop iteration
            for i in range(num_samples):
                prediction = np.dot(X[i], self.weights)
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

    def predict(self, X):
        predictions = np.dot(X, self.weights)
        return (predictions >= 0.5).astype(int)

# Train and Evaluate Widrow-Hoff
widrow_hoff = WidrowHoff(learning_rate=0.01, epochs=1000)
start_time = time.time()
widrow_hoff.fit(X_train_filtered, y_train_filtered)
runtime_wh = time.time() - start_time
y_pred_wh = widrow_hoff.predict(X_test)
accuracy_wh = accuracy_score(y_test, y_pred_wh)

print(f"Widrow-Hoff: Accuracy = {accuracy_wh * 100:.2f}%, Runtime = {runtime_wh:.4f} seconds")

# Classification Report for Widrow-Hoff
print("\nWidrow-Hoff Classification Report:")
print(classification_report(y_test, y_pred_wh, zero_division=1))
