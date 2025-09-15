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
        
        '''for i in wcss:
            if wcss.index(i) != 0:
                subtraction.append(i-i_prev)
            elif wcss.index(i) == 1:
                subtraction.append(i-i_prev)
            i_prev=i
        idx = np.argmin(subtraction) # місце найбільшого негативного вигину
        self.k_opt =idx+1 

        if len(inertias) >= 3:
            subtraction = np.diff(wcss, n=2)
            idx = np.argmin(subtraction)
            self.k_opt = idx + 1
        else:
            self.k_opt = ks[np.argmin(inertias)]
        return ks'''

    def silhouette_method(self):        
        silhouette_avg=[]
        #maxim=0
        for k in range(2, 11):
            kmeans = KMeans(k, random_state=42)
            labels = kmeans.fit_predict(self.data)

            sill_sc= silhouette_score(self.data, labels)
            '''if sill_sc>maxim:
                maxim=sill_sc
                maxim_k=k'''

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


        '''for k in range(1, 11):
            silhouette_avg = silhouette_score(self.data, self.labels_)

            print("For n_clusters =", k,", the average silhouette_score is :", silhouette_avg)'''
            
        '''silhouette_scores = []
        for k in range(1, 11):    
            score = silhouette_score(self.data, self.labels_)
            silhouette_scores.append(score)
            print(f"For n_clusters = {k}, the average silhouette score is: {score:.4f}")

        # Plotting the silhouette scores to find the optimal k
        plt.plot(range(1, 11), silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Analysis for Optimal k")
        plt.grid(True)
        plt.show()

        # The optimal k is typically the one with the highest silhouette score.
        self.k_opt = np.argmax(silhouette_scores) + 2 # +2 because range starts from 2
        print(f"Optimal number of clusters based on Silhouette Score: {self.k_opt}")

        return self.k_opt'''
        
        '''ks = list(range(k_min, k_max + 1))
        sil_scores = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.data)
            if len(set(labels)) == 1:
                sil_scores.append(-1)
            else:
                sil_scores.append(silhouette_score(self.data, labels))

        plt.figure()
        plt.plot(ks, sil_scores, marker="o")
        plt.xlabel("k")
        plt.ylabel("Average silhouette score")
        plt.title("Silhouette vs k")
        plt.grid(True)
        plt.show()

        self.k_opt = int(ks[np.argmax(sil_scores)])
        return ks, sil_scores'''

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

        '''rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        X = self.data
        n, d = X.shape
        ks = list(range(k_min, k_max + 1))
        log_wks = []
        log_wk_refs = {k: [] for k in ks}

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            Wk = km.inertia_
            log_wks.append(np.log(Wk + 1e-12))

            for b in range(B):
                X_ref = rng.uniform(low=mins, high=maxs, size=(n, d))
                km_ref = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                km_ref.fit(X_ref)
                log_wk_refs[k].append(np.log(km_ref.inertia_ + 1e-12))

        gaps = []
        sds = []
        for i, k in enumerate(ks):
            ref_vals = np.array(log_wk_refs[k])
            gap_k = np.mean(ref_vals) - log_wks[i]
            sd_k = np.sqrt(1.0 + 1.0 / B) * np.std(ref_vals, ddof=1)
            gaps.append(gap_k)
            sds.append(sd_k)

        # правило Tibshirani: обрати найменший k такий, що gap(k) >= gap(k+1) - s_{k+1}
        opt_k = ks[0]
        for i in range(len(ks) - 1):
            if gaps[i] >= gaps[i + 1] - sds[i + 1]:
                opt_k = ks[i]
                break
            else:
                opt_k = ks[-1]
        self.k_opt = int(opt_k)

        plt.figure()
        plt.errorbar(ks, gaps, yerr=sds, marker="o", linestyle="-")
        plt.xlabel("k")
        plt.ylabel("Gap(k)")
        plt.title("Gap statistic")
        plt.grid(True)
        plt.show()

        self._gap_results = {
            "ks": ks,
            "gaps": gaps,
            "sdk": sds,
            "log_wks": log_wks,
            "log_wk_refs": log_wk_refs,
        }
        return self._gap_results
        '''
    
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
        
        '''
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        X = np.asarray(self.data)
        n, d = X.shape
        m = min(sampling_size, n - 1)

        random_indices = rng.choice(n, size=m, replace=False)
        X_m = X[random_indices]

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        U = rng.uniform(low=mins, high=maxs, size=(m, d))

        nbrs = NearestNeighbors(n_neighbors=1).fit(X)

        u_distances = nbrs.kneighbors(U, return_distance=True)[0].ravel()

        nbrs2 = NearestNeighbors(n_neighbors=2).fit(X)
        w_dists_all = nbrs2.kneighbors(X_m, return_distance=True)[0]
        w_distances = w_dists_all[:, 1]

        u_power = u_distances ** d
        w_power = w_distances ** d

        H = u_power.sum() / (u_power.sum() + w_power.sum() + 1e-12)
        self.hopkins_statistic = float(H)
        return self.hopkins_statistic'''

    def _metric_calculation(self):
        centers, u, u0, d, jm, p, fPC = fuzz.cmeans(
            self.data.T, 3, m=2, error=0.005, maxiter=1000, init=None
        )

        #print('\nFPC ',fPC) # Fuzzy Partition Coefficient, Оцінює чіткість кластеризації, чим ближче до 1, тим краща сегментація
        
        unique_elements, counts = np.unique(self.data, return_counts=True)
        probabilities = counts / len(self.data)

        EC = scipy.stats.entropy(probabilities, base=2)
        #print('EC ', EC) # The higher. The particles are scattered randomly and are in constant, disordered motion

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

        #print('PI ',partition_indx)
        
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
        
        '''if labels is None:
            labels = self.labels_

        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        k = len(unique_labels)
        n = self.data.shape[0]

        # centers (якщо ще немає) — обчислимо
        if self.centers_ is None or len(self.centers_) != k:
            centers = []
            for lab in unique_labels:
                members = self.data[labels == lab]
                centers.append(members.mean(axis=0))
            centers = np.vstack(centers)
        else:
            centers = self.centers_

        # нечітка матриця U (аппроксимація)
        U = self._compute_fuzzy_membership(self.data, centers, m=m)
        # Partition Coefficient (PC)
        PC = np.sum(U ** 2) / n

        # Partition Entropy (PE)
        eps = 1e-12
        PE = - np.sum(U * np.log(U + eps)) / n

        # Xie-Beni index (часто називають partition index) —
        # XB = sum_{i,j} u_ij^m * ||x_i - v_j||^2  / (n * min_{p!=q} ||v_p - v_q||^2)
        n_samples = n
        # numerator
        distances_sq = cdist(self.data, centers, metric="sqeuclidean")  # (n,k)
        num = np.sum((U ** m) * distances_sq)
        # den
        if k >= 2:
            center_dists_sq = squareform(pdist(centers, metric="sqeuclidean"))
            # мінімальна міжцентрична відстань (поза діагоналлю)
            min_center_dist_sq = np.min(center_dists_sq[np.nonzero(center_dists_sq)])
            den = n_samples * (min_center_dist_sq + 1e-12)
            XB = num / den
        else:
            XB = np.nan

        # Separation index — реалізуємо через Dunn index:
        # Dunn = (min inter-cluster distance) / (max intra-cluster diameter)
        # точна реалізація: для кожної пари кластерів візьмемо мін відстаней між їх точками (inter),
        # для кожного кластеру — max відстань між парою точок (intra diameter).
        # Це O(n^2) — ок для невеликих наборів (Iris).
        max_intra = 0.0
        for lab in unique_labels:
            members = self.data[labels == lab]
            if members.shape[0] <= 1:
                diam = 0.0
            else:
                diam = np.max(pdist(members, metric="euclidean"))
            if diam > max_intra:
                max_intra = diam

        min_inter = np.inf
        for i, la in enumerate(unique_labels):
            for lb in unique_labels[i + 1:]:
                Xi = self.data[labels == la]
                Xj = self.data[labels == lb]
                if Xi.shape[0] == 0 or Xj.shape[0] == 0:
                    continue
                # мін відстаней між точками у різних кластерах
                d_ij = np.min(cdist(Xi, Xj, metric="euclidean"))
                if d_ij < min_inter:
                    min_inter = d_ij

        if max_intra == 0:
            dunn = np.inf
        else:
            dunn = min_inter / (max_intra + 1e-12)

        metrics = {
            "PC": float(PC),
            "PE": float(PE),
            "XieBeni": float(XB) if not np.isnan(XB) else None,
            "Dunn": float(dunn),
        }
        return metrics'''

class Mlflow_validator:
    def __init__(self, k_opt=0, hopk=0, metrics={'Metrix':0}):
        self.client=None
        self.k_opt=k_opt
        self.hopk=hopk
        self.metrics=metrics
        self.run_id=None
        self.experimen=None

        self.data={
            'K_opt':self.k_opt,
            'Hopk':self.hopk,
            'Metrics':self.metrics}
        
        with open("artifacts.txt", "w") as f:
            for key, value in self.data.items():
                f.write(f"{key} = {value}\n")


    def _initialization_mlflow(self):
        experiment_name = "New_Experiment",
        tracking_uri = "http://127.0.0.1:8000",
        run_name = "first_run"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)

        print('run id ', run.info.run_id)
        
        return run


    def _log_artifacts(self):
        self.client.log_artifact("artifacts.txt")
        #os.remove("artifacts.txt")
            
    def _download_artifact(self):
        local_path = download_artifacts(run_id=self.experiment.experiment_id, artifact_path="artifacts.txt")
        print("local_path", local_path)
        

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    ca = ClasterAnalysis(X, y, n_clusters=3, method="nearest_neighbour")
    #model = ca._clustering_procedure()
    #print('Model ', model)
    
    k_optim=ca.elbow_method()
    #k_optim=ca.silhouette_method()
    #k_optim=ca.gap_statistic()

    #print('k_opt ',k_optim)

    hopk=ca.hopkins_statistic()
    #print('Hopkins ', hopk)

    
    metrics = ca._metric_calculation()
    #print('Metrics ', metrics)

    mlf=Mlflow_validator(k_optim,hopk,metrics)
    mlf._initialization_mlflow()
    mlf._log_artifacts()
    mlf._download_artifact()


    

main()