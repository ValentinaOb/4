from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

class ClasterAnalysis:
    iris = load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=42, stratify=y
    )


    data=list(zip(x, y))

    n_clusters=None
    method='k_means' #nearest_neighbour
    #k_opt='elbow_method' #silhouette_method #gap_statistic
    Hopkins_statistic=0 #?

    def k_means(x,y, data, n_clusters):
        kmeans = KMeans(n_clusters)
        kmeans.fit(data)
        plt.scatter(x, y, c=kmeans.labels_)
        plt.show()

    def nearest_neighbour(x_train,y_train, x_test, y_test,x_scaled,y):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        # Predict and evaluate
        y_pred = knn.predict(x_test)
        print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

        k_range = range(1, 21)
        cv_scores = []

        # Evaluate each k using 5-fold cross-validation
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())

        # Plot accuracy vs. k
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, cv_scores, marker='o')
        plt.title("k-NN Cross-Validation Accuracy vs k")
        plt.xlabel("Number of Neighbors: k")
        plt.ylabel("Cross-Validated Accuracy")
        plt.grid(True)
        plt.show()

        # Best k
        best_k = k_range[np.argmax(cv_scores)]
        print(f"Best k from cross-validation: {best_k}")


        best_knn = KNeighborsClassifier(n_neighbors=best_k)
        best_knn.fit(x_train, y_train)

        # Predict on test data
        y_pred = best_knn.predict(x_test)

        plt.show()



 #   def elbow_method():

#    def silhouette_method():

  #  def gap_statistic():


    def _clustering_procedure(data, method, n_clusters):
        print()

    def _metric_calculation(n_clusters,data):
        k=n_clusters
        N=len(data)
        partition_coefficient=0 #
        entropy_coefficient=0 #
        partition_index=0 #
        m=2
        v=0 #
        separation_index=0 #

    def Hopkins_statistic_calculation():
        print()

class Mlflow_validator:
    def initialization_mlflow():
        print()
    def _log_artifacts():
        print()
    def _download_artifact():
        print()