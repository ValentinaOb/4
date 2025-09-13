from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

class ClasterAnalysis:
    x = [random.randint(0, 99) for _ in range(15)]
    y = [random.randint(0, 99) for _ in range(15)]

    data=list(zip(x, y))

    n_clusters=None
    method='k_means' #nearest_neighbour
    k_opt='elbow_method' #silhouette_method #gap_statistic
    Hopkins_statistic=0 #?

    def k_means(x,y, data, n_clusters):
        kmeans = KMeans(n_clusters)
        kmeans.fit(data)
        plt.scatter(x, y, c=kmeans.labels_)
        plt.show()

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