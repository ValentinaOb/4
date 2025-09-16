import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_samples
from pyclustertend import hopkins
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import skfuzzy as fuzz
from scipy.stats import entropy
import scipy.stats
from sklearn.metrics.pairwise import euclidean_distances
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
import os
import time

class ClasterAnalysis:

    def __init__(self, data, data_y, n_clusters=None, method="k_means", random_state=42):
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number]).values
        else:
            data = np.asarray(data, dtype=float)

        self.data = data
        self.data_y = data_y
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state

        self.k_opt = None
        self.hopkins = None
        self.labels_ = None
        #self.centers_ = None
        #self.inertia_ = None  # сумарна WCSS (для kmeans) або обчислена для інших методів
        #self._gap_results = None

    def _clustering_procedure(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters

        if self.method == "k_means":
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(self.data)

            self.labels_ = kmeans.labels_
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels_, cmap='viridis', s=50)
            plt.title("K-Means clustering on Iris dataset (2 features)")
            plt.xlabel("Sepal length")
            plt.ylabel("Sepal width")
            plt.grid(True)

            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
            plt.legend()

            plt.show()

            print('Centroids ', centroids)

            return kmeans
        
        elif self.method == "nearest_neighbour":
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(self.data, self.data_y)

            pred = knn.predict(self.data)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=pred, cmap='viridis', s=50)
            plt.title("KNN classification on Iris dataset (2 features)")
            plt.xlabel("Sepal length")
            plt.ylabel("Sepal width")
            plt.grid(True)
            plt.show()

            return knn

    def elbow_method(self):
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)

        subtraction=[]
        for i in wcss:
            if wcss.index(i) > 1:
                subtraction.append(i-i_prev)
            elif wcss.index(i) == 1:
                subtraction.append(i-i_prev)
            i_prev=i
            
        idx = np.argmin(subtraction)
        self.k_opt = idx + 2        
        
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.show()

        
        return self.k_opt
        # Метод Ліктя. Коли верхні і нижні лінії сходяться максимально один до одного - саме таку к-ть кластерів і рекомендується
        
    def silhouette_method(self):        
        silhouette_avg=[]
        for k in range(2, 11):
            kmeans = KMeans(k, random_state=42)
            labels = kmeans.fit_predict(self.data)

            sill_sc= silhouette_score(self.data, labels)
            silhouette_avg.append(silhouette_score(self.data, labels))
            print("For n_clusters ", k,", the average silhouette_score is ", sill_sc)
        k_opt = np.argmax(np.diff(silhouette_avg))

        plt.plot(range(2, 11), silhouette_avg, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Method")
        plt.grid(True)
        plt.show()
        
        return k_opt

    def gap_statistic(self):
        gap_values = []
        original_inertia=[]
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            original_inertia = kmeans.inertia_

            reference_inertia = []
            for _ in range(1,11):
                random_data = np.random.uniform(low=self.data.min(axis=0), high=self.data.max(axis=0), size=self.data.shape)
                kmeans.fit(random_data)
                reference_inertia.append(kmeans.inertia_)

            gap = np.log(np.mean(reference_inertia)) - np.log(original_inertia)
            gap_values.append(gap)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), gap_values, marker='o')
        plt.title('Gap Statistic')
        plt.xlabel('Number of Clusters k')
        plt.ylabel('Gap Statistic')
        plt.grid()
        plt.show()
        
        k_opt = np.argmax(np.diff(gap_values)) + 2
        print('l ',np.diff(gap_values))
        print(gap_values)

        return k_opt

    def hopkins_statistic(self):
        self.hopkins = hopkins(self.data, self.data.shape[0])

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data)
        statistic_minmax = hopkins(scaled_data, scaled_data.shape[0])
        print('Hopkins statistic. MinMax Scaling ', statistic_minmax)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        statistic_standard = hopkins(scaled_data, scaled_data.shape[0])
        print('Hopkins statistic. Standard Scaling', statistic_standard)

        return self.hopkins
    
        #MinMax Scaling is ideal when your features have known and consistent ranges, and you're using algorithms sensitive to the magnitude of features
        #Standard Scalingis. data has different units or magnitudes. It helps to ensure that each feature contributes equally to distance-based metrics used in clustering algorithms, making it a safer choice for clustering
    
        #Values close to 0: Indicate that the data is regularly spaced, also suggesting a lack of strong clustering tendency
        #Values around 0.5: Suggest that the data is randomly distributed and unlikely to contain significant clusters
        #Values close to 1: Indicate that the data has a high tendency to cluster, suggesting the presence of meaningful clusters

    def _metric_calculation(self):
        centers, u, u0, d, jm, p, fPC = fuzz.cmeans(
            self.data.T, 3, m=2, error=0.005, maxiter=1000, init=None
        )
        
        unique_elements, counts = np.unique(self.data, return_counts=True)
        probabilities = counts / len(self.data)

        EC = scipy.stats.entropy(probabilities, base=2)
        
        # PI
        U = u.T  # матриця приналежностей
        N, C = U.shape
        numerator = 0
        m=2
        for i in range(N):
            for j in range(C):
                dist_sq = np.linalg.norm(self.data[i] - centers[j])**2
                numerator += (U[i, j]**m) * dist_sq

        min_center_dist_sq = np.min([
            np.linalg.norm(centers[i] - centers[j])**2
            for i in range(C) for j in range(C) if i != j
        ])

        partition_indx= numerator / (N * min_center_dist_sq) # Враховує як компактність всередині кластерів, так і розділення відстань між центрами кластерів, < — краще
        
        #SI
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.data)

        centroids = kmeans.cluster_centers_
        unique_labels = np.unique(self.labels_)
        
        centroid_distances = euclidean_distances(centroids, centroids)
        total_separation = 0
        for i in range(len(unique_labels)):
            distances_from_centroid_i = np.delete(centroid_distances[i], i)
            total_separation += np.mean(distances_from_centroid_i)

        SI= total_separation / len(unique_labels)
        #print('SI ', SI) #Оцінює наскільки добре розділені кластери, > — краще

        metrics = {
            'PC': fPC,
            'EC': EC,
            'PI': partition_indx,
            'SI': SI
        }
        return metrics
        
class Mlflow_validator:
    def __init__(self, method, df, k_opt=0, hopk=0, metrics={'Metrix':0}):
        self.client=None
        self.method=method
        self.df=df
        self.k_opt=k_opt
        self.hopk=hopk
        self.metrics=metrics

        self.run_id=None
        #self.experiment=None
        
        self.url="http://127.0.0.1:8080"


    def _initialization_mlflow(self):        
        mlflow.set_tracking_uri(self.url)
        mlflow.set_experiment("MLflow Quickstart")
        self.run_id = mlflow.start_run().info.run_id

        self._log_artifacts()
        self._download_artifact()
        self.end_run()

    def _log_artifacts(self):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("method", self.method)
        
        numb_obj, columns = self.df.shape
        mlflow.log_param("numb_obj", numb_obj)
        mlflow.log_param("columns", columns)
        mlflow.log_param("k_opt", self.k_opt)

        for key in self.metrics:
            mlflow.log_metric(key,self.metrics.get(key))

        mlflow.log_artifact("ML/pima.csv")
        #os.remove("pima.csv")
            
    def _download_artifact(self):
        local_path_file = download_artifacts(artifact_uri="ML/pima.csv",dst_path="KA/")
        print(f"Specific artifact downloaded to: {local_path_file}")

    def end_run(self):
        mlflow.end_run()

def main():
    # mlflow server --host 127.0.0.1 --port 8080 
    
    df = pd.read_csv("ML/pima.csv")  
    X = df 
    y = df['Age']  

    method="k_means"

    ca = ClasterAnalysis(X, y, n_clusters=2, method=method)
    #k_optim=ca.elbow_method()
    #k_optim=ca.silhouette_method()
    k_optim=ca.gap_statistic()

    print('k_opt ',k_optim)

    hopk=ca.hopkins_statistic()
    print('Hopkins ', hopk)

    
    metrics = ca._metric_calculation()
    print('Metrics ', metrics)

    mlf=Mlflow_validator(method,df,k_optim,hopk,metrics)
    mlf._initialization_mlflow()
    


    

main()