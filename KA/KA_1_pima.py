import glob
import math
import random
from matplotlib import cm
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pyclustertend import hopkins
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import skfuzzy as fuzz
import scipy.stats
from sklearn.metrics.pairwise import euclidean_distances
from mlflow.artifacts import download_artifacts
import os
import time
from sklearn.mixture import GaussianMixture
import networkx as nx
from networkx.algorithms.community import girvan_newman
import markov_clustering as mc
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs
import matplotlib as mpl
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.metrics import pairwise_distances_argmin

class ClasterAnalysis:

    def __init__(self, data, data_y, n_clusters=None, method="k_means",use=1, random_state=42):
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
        self.numb_of_chunks=None
        self.use=use
        #self.centers_ = None
        #self.inertia_ = None  # сумарна WCSS (для kmeans) або обчислена для інших методів
        #self._gap_results = None

    '''def __init__(self, G=None, n_clusters=None, method="edge_cut"):
        self.G = G
        self.n_clusters=n_clusters
        self.method=method
        self.clusters=None
        self.G_new=None
        self.compat_matrix=None
        self.tochastic_matrix=None
        self.k_opt=None
        self.eigvals=None
        self.A=None
        self.labels=[]
    '''
    def _clustering_procedure(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters

        if self.method == "k_means":
            if self.G:
                pos = nx.spring_layout(self.G, seed=42)
                X = np.array([pos[node] for node in self.G.nodes()]) #координати вузлів у матрицю ознак

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(self.data)
            self.labels_ = kmeans.labels_
            
            plt.figure(figsize=(8, 6))

            if self.G:
                colors = ['lightblue', 'lightgreen', 'salmon', 'yellow', 'orange', 'violet']
                for i in range(4):
                    cluster_nodes = [node for node, label in zip(self.G.nodes(), self.labels_) if label == i]
                    nx.draw_networkx_nodes(self.G, pos, nodelist=cluster_nodes, node_color=colors[i])
                nx.draw_networkx_edges(self.G, pos, alpha=0.5)

            else:
                plt.scatter(self.data[:, 4], self.data[:, 5], c=self.labels_, cmap='viridis', s=50)
                plt.xlabel("DiabetesPedigreeFunction")
                plt.ylabel("Age")
                plt.grid(True)

                centroids = kmeans.cluster_centers_
                plt.scatter(centroids[:, 4], centroids[:, 5], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
            
            plt.title("K-Means clustering")
            plt.savefig("KA/k_means.jpg", dpi=300, bbox_inches='tight') 
            plt.show()
            #print('Centroids ', centroids[:,4])

            return kmeans
        
        elif self.method == "nearest_neighbour":

            knn = KNeighborsClassifier(n_neighbors=self.n_clusters)
            knn.fit(self.data, self.data_y) 

            pred = knn.predict(self.data)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 4], self.data[:, 5], c=pred, cmap='viridis', s=50)
            plt.title("KNN classification (2 features)")
            plt.xlabel("DiabetesPedigreeFunction")
            plt.ylabel("Age")
            plt.grid(True)
            plt.show()

            return knn

        elif self.method == "c_means":
            centers, u, u0, d, jm, p, fpc = fuzz.cmeans(
            self.data.T, n_clusters, m=2, error=0.005, maxiter=1000, init=None
            )
            
            hard_clusters = np.argmax(u, axis=0)

            plt.figure(figsize=(8, 6))
            plt.scatter(
                self.data[:, 4], self.data[:, 5],
                c=hard_clusters, cmap='viridis', s=50
            )
            plt.title("C-Means clustering (2 features)")
            plt.xlabel("DiabetesPedigreeFunction")
            plt.ylabel("Age")
            plt.grid(True)

            plt.scatter(
                centers[:, 4], centers[:, 5],
                c='red', s=200, alpha=0.75, marker='X', label='Centers'
            )
            plt.legend()
            plt.show()

        elif self.method == "gath_geva":
            
            gg = GaussianMixture(n_components=n_clusters)
            gg.fit(self.data, self.data_y)
            gg_membership = gg.predict_proba(self.data)   # fuzzy матриця U
            gg_labels = np.argmax(gg_membership, axis=1)

            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 4], self.data[:, 5], c=gg_labels, cmap='viridis', s=50)
            plt.title("Gath_Geva (2 features, fuzzy membership)")
            plt.xlabel("DiabetesPedigreeFunction")
            plt.ylabel("Age")
            plt.grid(True)
            plt.scatter(gg.means_[:, 4], gg.means_[:, 5], c="red", marker="X", s=200, alpha=0.75, label='Centroids')
            plt.show()
            
            #замість "жорстких" міток (predict) - ймовірності приналежності (predict_proba), "fuzzy" членство
        
        elif self.method == "gustafson_kessel":

            gk = GaussianMixture(n_components=n_clusters)
            gk_labels = gk.fit_predict(self.data)

            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 4], self.data[:, 5], c=gk_labels, cmap="viridis", s=40)
            plt.scatter(gk.means_[:, 4], gk.means_[:, 5], c="red", marker="X", s=200)
            plt.title("Gustafson_Kessel (GaussianMixture)")
            plt.show()
            
            # прац з еліпсоїдними кластерами (відповідає GK)

        elif self.method == "edge_cut":
            G_copy = self.G.copy()
            while nx.number_connected_components(G_copy) < n_clusters:
                edge_bet = nx.edge_betweenness_centrality(G_copy) # Розрах центральності ребер
                edge_to_remove = max(edge_bet, key=edge_bet.get)
                G_copy.remove_edge(*edge_to_remove)

            self.clusters = [list(c) for c in nx.connected_components(G_copy)]
            self.G_new=G_copy

        elif self.method == "girvan_newman":
            G_copy = self.G.copy()
            communities_generator = girvan_newman(self.G)   # Генератор розбиттів
            self.clusters = next(communities_generator)
            self.clusters = [list(c) for c in self.clusters]
            edges_to_remove = []
            for u, v in G_copy.edges():
                for cluster in self.clusters:
                    if (u in cluster) and (v not in cluster):
                        edges_to_remove.append((u, v))
                        break

            G_copy.remove_edges_from(edges_to_remove)
            self.G_new=G_copy

        elif self.method == "markov_alg":
            matrix = csr_matrix(nx.to_scipy_sparse_array(self.G, dtype=float)) # розріджену матрицю суміжності
            result = mc.run_mcl(matrix, expansion=2, inflation=2)

            self.clusters = mc.get_clusters(result)
            self.clusters = [list(cluster) for cluster in self.clusters]

            G_copy = self.G.copy()
            edges_to_remove = []
            for u, v in G_copy.edges():
                for cluster in self.clusters:
                    if (u in cluster) and (v not in cluster):
                        edges_to_remove.append((u, v))
                        break
            G_copy.remove_edges_from(edges_to_remove)

            self.G_new=G_copy
        
        elif self.method == "pagerank_alg":
            threshold=0.03
            pr = nx.pagerank(self.G)
            
            high_nodes = [n for n, v in pr.items() if v >= threshold]
            low_nodes = [n for n, v in pr.items() if v < threshold]
            
            self.clusters = [high_nodes, low_nodes]
            
            G_copy = self.G.copy()
            edges_to_remove = []
            for u, v in G_copy.edges():
                if (u in high_nodes and v in low_nodes) or (v in high_nodes and u in low_nodes):
                    edges_to_remove.append((u, v))
            G_copy.remove_edges_from(edges_to_remove)
            
            self.G_new=G_copy

        elif self.method == "based_on_repr":
            N = len(self.data)
            alpha=0.2
            
            n_repr = int(alpha * N)
            repr_idx = np.random.choice(N, n_repr, replace=False)
            data_repr = self.data[repr_idx]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data_repr)
            
            labels = pairwise_distances_argmin(self.data, kmeans.cluster_centers_)

            plt.figure(figsize=(8, 6))    
            plt.scatter(self.data[:, 4], self.data[:, 5], c=labels, s=25, cmap='tab10', alpha=0.6, label="Objects")
            plt.scatter(self.data[repr_idx, 4], self.data[repr_idx, 5], 
                        c='grey', s=60, edgecolors='white', marker='o', label='Representatives')
            plt.scatter(kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 5],
                        c='red', s=100, marker='X', edgecolors='black', label='Centers')
            
            plt.title("Based on Representatives")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig("KA/based_on_repr.jpg", dpi=300, bbox_inches='tight') 
            plt.show()

        elif self.method == "bfr":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels_kmeans = kmeans.fit_predict(self.data)

            final_labels = labels_kmeans.copy()

            M_list = []
            D_list = []
            for i in range(self.n_clusters):
                pts_idx = np.where(labels_kmeans == i)[0]
                cluster_points = self.data[pts_idx]
                M = np.mean(cluster_points, axis=0)
                D = np.var(cluster_points, axis=0, ddof=1)
                M_list.append(M)
                D_list.append(D)

            t = 12
            m = 2
            use = 1

            other_idx = []

            for idx in range(len(self.data)):
                x = self.data[idx]
                assigned = False

                for i in range(self.n_clusters):
                    M = M_list[i]
                    D = D_list[i]

                    if use == 1:
                        d = np.sqrt(np.sum(((x - M) / np.sqrt(D)) ** 2))
                        if d <= t:
                            final_labels[idx] = i
                            M_new = (M + x) / 2.0
                            D_new = (D + (x - M) ** 2) / 2.0
                            M_list[i] = M_new
                            D_list[i] = D_new
                            assigned = True
                            break
                    else:
                        lower = M - m * np.sqrt(D)
                        upper = M + m * np.sqrt(D)
                        if np.all((x >= lower) & (x <= upper)):
                            final_labels[idx] = i
                            M_new = (M + x) / 2.0
                            D_new = (D + (x - M) ** 2) / 2.0
                            M_list[i] = M_new
                            D_list[i] = D_new
                            assigned = True
                            break

                if not assigned:
                    final_labels[idx] = -1
                    other_idx.append(idx)

            if len(other_idx) > 0:
                remaining = []
                for idx in other_idx:
                    x = self.data[idx]
                    assigned = False
                    for i in range(self.n_clusters):
                        M = M_list[i]
                        D = D_list[i]
                        d = np.sqrt(np.sum(((x - M) / np.sqrt(D)) ** 2))
                        if d <= t:
                            final_labels[idx] = i
                            M_new = (M + x) / 2.0
                            D_new = (D + (x - M) ** 2) / 2.0
                            M_list[i] = M_new
                            D_list[i] = D_new
                            assigned = True
                            break
                    if not assigned:
                        remaining.append(idx)

            for i in range(self.n_clusters):
                pts_idx = np.where(final_labels == i)[0]
                print('\nCl ',i+1, len(pts_idx))
                print('\nM ', np.round(M_list[i][4:6], 3), '\nD ', np.round(D_list[i][4:6], 3))

            unassigned_idx = np.where(final_labels == -1)[0]
            print('\nUnassigned ', len(unassigned_idx))
            if len(unassigned_idx) > 0:
                print(unassigned_idx[:6])

            colors = ['lightblue', 'lightgreen', 'salmon', 'yellow', 'orange', 'violet']
            for i in range(self.n_clusters):
                idxs = np.where(final_labels == i)[0]
                if len(idxs) > 0:
                    pts = self.data[idxs]
                    plt.scatter(pts[:, 4], pts[:, 5], color=colors[i % len(colors)], label=f'Cl {i+1}', alpha=0.6)

            if len(unassigned_idx) > 0:
                plt.scatter(self.data[unassigned_idx, 4], self.data[unassigned_idx, 5], s=60, label='Unassigned')

            for i in range(self.n_clusters):
                plt.scatter(M_list[i][4], M_list[i][5], color='k', marker='x', s=100)

            plt.title('BFR')
            plt.xlabel('DiabetesPedigreeFunction')
            plt.ylabel('Age')
            plt.legend()
            plt.savefig("KA/bfr.jpg", dpi=300, bbox_inches='tight') 
            plt.show()
        
        elif self.method == "cure":
            N = len(self.data)
            alpha=0.2
            
            n_repr = int(alpha * N)
            repr_idx = np.random.choice(N, n_repr, replace=False)
            data_repr = self.data[repr_idx]

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            labels_repr = model.fit_predict(data_repr)
            centers = np.array([data_repr[labels_repr == i].mean(axis=0) for i in range(n_clusters)])
            labels = pairwise_distances_argmin(self.data, centers)
  
            plt.scatter(self.data[:, 4], self.data[:, 5], c=labels, s=25, cmap='tab10', alpha=0.6, label="Objects")
            plt.scatter(self.data[repr_idx, 4], self.data[repr_idx, 5], 
                        c='grey', s=60, edgecolors='white', marker='o', label='Representatives')
            plt.scatter(centers[:, 4], centers[:, 5],
                        c='red', s=100, marker='X', edgecolors='black', label='Centers')
            
            plt.title("CURE")
            plt.legend()
            plt.savefig("KA/cure.jpg", dpi=300, bbox_inches='tight') 
            plt.show()

        elif self.method == "birch":
            N = len(self.data)
            alpha=0.2
            
            n_repr = int(alpha * N)
            repr_idx = np.random.choice(N, n_repr, replace=False)
            data_repr = self.data[repr_idx]

            model = Birch(n_clusters=n_clusters, threshold=0.5)
            labels = model.fit_predict(self.data)

            plt.figure(figsize=(8, 6))    
            plt.scatter(self.data[:, 4], self.data[:, 5], c=labels, s=25, cmap='tab10', alpha=0.6, label="Objects")
            plt.scatter(self.data[repr_idx, 4], self.data[repr_idx, 5], c='grey', s=60, edgecolors='white', marker='o', label='Representatives')
            
            plt.title("Birch")
            plt.legend()
            plt.savefig("KA/birch.jpg", dpi=300, bbox_inches='tight') 
            plt.show()

        elif self.method == "dbscan":
            scale_data = MinMaxScaler().fit_transform(self.data)

            dbscan = DBSCAN(eps=0.8, min_samples=5)
            labels = dbscan.fit_predict(scale_data)

            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l != -1])

            plt.figure(figsize=(8, 6))
            colors = plt.cm.get_cmap('tab10', n_clusters)
            for k in unique_labels:
                if k == -1:
                    col = 'lightgray'
                    label = 'Noise'
                else:
                    col = colors(k)
                    label = f'Cluster {k}'
                
                mask = (labels == k)
                plt.scatter(scale_data[mask, 4], scale_data[mask, 5], c=[col], s=30,alpha=0.7, label=label)

            plt.title("DBSCAN")
            plt.legend()
            plt.savefig("KA/dbscanjpg", dpi=300, bbox_inches='tight') 
            plt.show()
            
    def _graph_characteristics(self):
        self.compat_matrix=nx.to_scipy_sparse_array(self.G, dtype=float)

        if self.compat_matrix.shape[0] != self.compat_matrix.shape[1]:
            raise ValueError("Input matrix must be square.")
        if (self.compat_matrix.toarray() < 0).any():
            raise ValueError("All elements of the input matrix must be non-negative.")

        row_sums = self.compat_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.stochastic_matrix = self.compat_matrix / row_sums

    def _optimal_clusters_number(self):
        N = self.stochastic_matrix.shape[0]
        h=2
        eigvals = eigs(self.stochastic_matrix, k=min(50, N-2), sigma=1.0, return_eigenvectors=False) # власні значення

        radius = h / np.sqrt(N)
        k_opt = np.sum(np.abs(eigvals - 1) < radius)
        print('k_opt ', k_opt)

        plt.figure(figsize=(6,6))
        plt.gca().add_patch(plt.Circle((0,0), 1, color='gray', alpha=0.3))
        plt.gca().add_patch(plt.Circle((1,0), radius, color='skyblue', alpha=0.5))
        plt.scatter(eigvals.real, eigvals.imag, color='red')
        plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Eigenvalues of a matrix P. k_opt={k_opt}')
        plt.xlabel('Re(λ)'); plt.ylabel('Im(λ)')
        plt.savefig("KA/opt_clust_numb.jpg", dpi=300, bbox_inches='tight') 
        plt.show()

        self.k_opt, self.eigvals=k_opt, eigvals
        
    def graph_create(self,K, N, a1, b1, a2, b2):
        nodes = N // K
        self.G = nx.Graph()
        rng = np.random.default_rng()
        self.G.add_nodes_from(range(N))

        # ств зв’язків
        for i in range(N):
            for j in range(i + 1, N):
                ci = i // nodes
                cj = j // nodes

                if ci == cj:
                    p = rng.uniform(a2, b2)
                else:
                    p = rng.uniform(a1, b1)

                if rng.random() < p:
                    self.G.add_edge(i, j)

        self.method='markov_alg'
        self._clustering_procedure(K)
        
    def metrix_graph(self):
        comps = [self.G_new.subgraph(c).copy() for c in nx.connected_components(self.G_new)]
        for i, comp in enumerate(comps):
                print(f"Comp {i+1}   {comp.number_of_nodes()} nodes    |    {comp.number_of_edges()} edges")

        grade_num = len(comps)
        print('Grade clust numb ', grade_num)

        if nx.is_connected(self.G):
            r = nx.radius(self.G)
            d = nx.diameter(self.G)

        diameters = []
        for i, c in enumerate(comps, 1):
            diameters.append(nx.diameter(c))
            print(f"Comp {i}    d - {nx.diameter(c) if len(c)>1 else 0}")

        max_diam = max(diameters) if diameters else 0
        print('Max d subgraph', max_diam)

        metrics = {
            'GN': grade_num,
            'R': r,
            'D': d,
            'Max D': max_diam
        }

        return metrics

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        pos = nx.spring_layout(self.G, seed=42)

        nx.draw(self.G, pos, with_labels=True, ax=axes[0])
        axes[0].set_title("Initial graph")

        nx.draw(self.G_new, pos, with_labels=True, ax=axes[1])
        axes[1].set_title(self.method)
        plt.show()

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
        plt.savefig("KA/elbow_method.jpg", dpi=300, bbox_inches='tight') 
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
        plt.savefig("KA/silhouette_method.jpg", dpi=300, bbox_inches='tight') 
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
        plt.savefig("KA/gap_statistic.jpg", dpi=300, bbox_inches='tight') 
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
        
    def split_into_chunks(self, size):
        for i in range(0, len(self.data), size):
            chunks=self.data[i:i + size]
            print('Chunks ', chunks, '\n numb ', len(chunks))
        self.numb_of_chunks=len(chunks)            

class Mlflow_validator:
    '''def __init__(self, name,method, df, k_opt=0, hopk=0, metrics={'metrics':0}):
        self.client=None
        self.method=method
        self.df=df
        self.k_opt=k_opt
        self.hopk=hopk
        self.metrics=metrics

        self.run_id=None
        #self.experiment=None
        self.numb=0
        self.name=name
        
        self.url="http://127.0.0.1:8080"
    '''
    ''' def __init__(self,name, method, K,N,metrics):
        self.client=None
        self.method=method
        self.N=N
        
        self.K=K
        self.metrics=metrics
        
        self.run_id=None
        #self.experiment=None
        self.numb=1
        self.name=name

        self.url="http://127.0.0.1:8080"
    '''
    def __init__(self, name,method, df, k_opt=0, hopk=0, chunks=0, metrics={'metrics':0}):
        self.client=None
        self.method=method
        self.df=df
        self.k_opt=k_opt
        self.hopk=hopk
        self.metrics=metrics
        self.chunks=chunks

        self.run_id=None
        #self.experiment=None
        self.numb=0
        self.name=name
        
        self.url="http://127.0.0.1:8080"
   
    def _initialization_mlflow(self):        
        mlflow.set_tracking_uri(self.url)
        mlflow.set_experiment("MLflow Quickstart")
        self.run_id = mlflow.start_run(run_name=self.name).info.run_id

        self._log_artifacts(self.numb)
        self._download_artifact()
        self.end_run()

    def _log_artifacts(self, numb):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("method", self.method)

        if numb==0:
            '''numb_obj, columns = self.df.shape
            mlflow.log_param("numb_obj", numb_obj)
            mlflow.log_param("columns", columns)'''
            mlflow.log_param("k_opt", self.k_opt)
        else:
            mlflow.log_param("Nodes", self.N)
            mlflow.log_param("Clusters", self.K)

        for key in self.metrics:
            mlflow.log_metric(key,self.metrics.get(key))

        if numb==0:
            mlflow.log_artifact("ML/pima.csv",artifact_path="dataframe")
            mlflow.log_param("Hopkins", self.hopk)
            mlflow.log_param("Numb", self.numb)

        
        files=glob.glob("KA/*.jpg")
        for file in files:
            mlflow.log_artifact(file,artifact_path="features")
        #os.remove("pima.csv")
            os.remove(file)
            
    def _download_artifact(self):
        local_path_file = download_artifacts(artifact_uri="ML/pima.csv",dst_path="KA/")
        print(f"Specific artifact downloaded to: {local_path_file}")

    def end_run(self):
        mlflow.end_run()

def main():
    # mlflow server --host 127.0.0.1 --port 8080 
    
    df = pd.read_csv("ML/pima.csv")  
    df = df.apply(lambda col: col.fillna(round(col.mean(),3)))

    X = df 
    y = df['Outcome']  

    #method="k_means"
    #method="nearest_neighbour"
    #method="c_means"
    #method="gath_geva"
    #method="gustafson_kessel"

    method="based_on_repr"
    #method="bfr"
    #method="cure"
    #method="birch"
    #method="dbscan"

    ca = ClasterAnalysis(X, y, n_clusters=3, method=method, use=1)
    ca.split_into_chunks(5)

    ca._clustering_procedure()
    hopk=ca.hopkins_statistic()
    print('Hopkins ', hopk)
    metrics = ca._metric_calculation()
    print('Metrics ', metrics)
    
    mlf=Mlflow_validator(method,df,ca.k_opt,hopk, ca.numb_of_chunks, metrics)
    mlf._initialization_mlflow()


    #k_optim=ca.elbow_method()
    #k_optim=ca.silhouette_method()
    '''k_optim=ca.gap_statistic()

    print('k_opt ',k_optim)

    hopk=ca.hopkins_statistic()
    print('Hopkins ', hopk)

    
    metrics = ca._metric_calculation()
    print('Metrics ', metrics)

    name='ca'
    mlf=Mlflow_validator(name,method,df,k_optim,hopk,metrics)
    mlf._initialization_mlflow()'''

    #pagerank_scores = nx.pagerank(G, alpha=0.85)
    #print(pagerank_scores)

    '''   ! Graph'''
    
    #G = nx.karate_club_graph()     
    #method ='edge_cut'
    #method ='girvan_newman'
    #method ='markov_alg'
    #method ='pagerank_alg'
    

    #ca = ClasterAnalysis(G, n_clusters=3, method=method)
    #ca._graph_characteristics()
    
    #print('Stochastic matrix ',ca.stochastic_matrix)
    #print('Compatibility matrix ',ca.compat_matrix)

    '''ca._optimal_clusters_number()

    ca._clustering_procedure()
    for i, c in enumerate(ca.clusters, 1):
        print(f"Cluster {i}: {c}")
    ca.plot()
    ca.method='k_means'

    pos = nx.spring_layout(ca.G_new, seed=42)
    ca.data= np.array([pos[node] for node in ca.G_new.nodes()])
    ca._clustering_procedure(ca.k_opt)
    metrics = ca.metrix_graph()
    
    name='ca'
    mlf=Mlflow_validator(name,method,ca.G.number_of_nodes(),ca.k_opt,metrics)
    mlf._initialization_mlflow()
    
    #   
    print('\n\nCa1')
    K = 3
    N=1000
    a1, b1 = 0, 0.3   # між кластерами (слабкі зв’язки)
    a2, b2 = 0.7, 1.0 # всередині (сильні)
    
    ca1=ClasterAnalysis(n_clusters=K)
    ca1.graph_create(K, N, a1, b1, a2, b2)
    metrics= ca1.metrix_graph()
    name='ca1'
    mlf=Mlflow_validator(name,method,N,K,metrics)
    mlf._initialization_mlflow()

    #
    print('\n\nCa2')
    K=10
    a1, b1 = 0, 1
    a2, b2 = 0, 10
    ca2=ClasterAnalysis(n_clusters=K)
    ca2.graph_create(K, N, a1, b1, a2, b2)
    metrics=ca2.metrix_graph()
    name='ca2'
    mlf=Mlflow_validator(name,method,N,K,metrics)
    mlf._initialization_mlflow()


    print('\n\nCa3')
    K=10
    a1, b1 = 0, 1
    a2, b2 = 0, 2
    ca3=ClasterAnalysis(n_clusters=K)
    ca3.graph_create(K, N, a1, b1, a2, b2)
    metrics=ca3.metrix_graph()
    name='ca3'
    mlf=Mlflow_validator(name,method,N,K,metrics)
    mlf._initialization_mlflow()


    print('\n\nCa4')
    K=100
    a1, b1 = 0, 1
    a2, b2 = 0, 2
    ca4=ClasterAnalysis(n_clusters=K)
    ca4.graph_create(K, N, a1, b1, a2, b2)
    metrics=ca4.metrix_graph()
    name='ca4'
    mlf=Mlflow_validator(name,method,N,K,metrics)
    mlf._initialization_mlflow()
    '''

main()