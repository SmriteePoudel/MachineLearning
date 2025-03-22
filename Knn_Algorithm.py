import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        """Initialize KNN classifier with k neighbors"""
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store the training data"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Predict the class for each sample in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict the class for a single sample"""
        # Calculate distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        
        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Example usage with Iris dataset
def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the KNN classifier
    knn = KNN(k=3)
    knn.fit(X_train, y_test)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Visualize results for the first two features
    plt.figure(figsize=(10, 6))
    
    # Plot training points
    for i in range(3):
        plt.scatter(
            X_train[y_train == i, 0],
            X_train[y_train == i, 1],
            label=f'Class {i} (train)',
            alpha=0.5
        )
    
    # Plot test points
    for i in range(3):
        plt.scatter(
            X_test[y_test == i, 0],
            X_test[y_test == i, 1],
            marker='x',
            label=f'Class {i} (test)',
            alpha=0.7
        )
    
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('KNN Classification Results (First Two Features)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
