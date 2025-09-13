import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt


class ClasterAnalysis:

    def __init__(self, data, n_clusters=None, method="k_means", random_state=42):
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number]).values
        else:
            data = np.asarray(data, dtype=float)

        self.data = data
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state

        self.k_opt = None
        self.hopkins_statistic = None
        self.labels_ = None
        self.centers_ = None
        self.inertia_ = None  # сумарна WCSS (для kmeans) або обчислена для інших методів
        self._gap_results = None

    def _clustering_procedure(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters

        if self.method == "k_means":
            model = KMeans(n_clusters, random_state=self.random_state, n_init=10)
            model.fit(self.data)
            self.labels_ = model.labels_
            self.centers_ = model.cluster_centers_
            self.inertia_ = model.inertia_
            return model

        elif self.method == "nearest_neighbour":
            # single-link - найближчий сусід
            model = AgglomerativeClustering(n_clusters, linkage="single")
            self.labels_ = model.fit_predict(self.data)
            # середнє по кластеру (для подальших метрик)
            centers = []
            for k in range(n_clusters):
                members = self.data[self.labels_ == k]
                if len(members) == 0:
                    centers.append(np.zeros(self.data.shape[1]))
                else:
                    centers.append(np.mean(members, axis=0))
            self.centers_ = np.vstack(centers)
            # inertia як сума квадратів відхилень від центрів
            inertia = 0.0
            for k in range(n_clusters):
                members = self.data[self.labels_ == k]
                if len(members) > 0:
                    inertia += np.sum((members - self.centers_[k]) ** 2)
            self.inertia_ = inertia
            return model

    def elbow_method(self, k_min=1, k_max=10):
        ks = list(range(k_min, k_max + 1))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(self.data)
            inertias.append(km.inertia_)

        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k")
        plt.ylabel("WCSS / inertia")
        plt.title("Elbow plot")
        plt.grid(True)
        plt.show()

        # простий евристичний вибір "повороту" — за другою похідною (шукаємо найбільший вигин)
        if len(inertias) >= 3:
            second_deriv = np.diff(inertias, n=2)
            # місце найбільшого негативного вигину -> індекс у second_deriv
            idx = np.argmin(second_deriv)
            k_guess = ks[idx + 1]  # +1 через зсув другої похідної
            self.k_opt = int(k_guess)
        else:
            self.k_opt = ks[np.argmin(inertias)]
        return ks, inertias

    def silhouette_method(self, k_min=2, k_max=10):
        ks = list(range(k_min, k_max + 1))
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
        return ks, sil_scores

    def gap_statistic(self, k_min=1, k_max=10, B=10, random_state=None):
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
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

    def hopkins_statistic_calculation(self, sampling_size=50, random_state=None):
        """
        Обчислює статистику Хопкінса (значення ≈ 0.5 -> випадкова, ближче до 1 -> сильна кластеризованість).
        """
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
        return self.hopkins_statistic

    def _compute_fuzzy_membership(self, data, centers, m=2, eps=1e-8):
        """
        Якщо кластеризація 'жорстка' (labels_), приблизно побудуємо 'м' матрицю приналежності
        на основі відстаней до центрів (інверсія відстаней)
        m: розмірність нечіткої експоненти (звично m=2).
        Повертає U матрицю (n_samples x k).
        """
        D = cdist(data, centers, metric="euclidean")  # (n, k)
        # щоб уникнути ділення на нуль:
        inv = 1.0 / (D + eps)
        U = inv / np.sum(inv, axis=1, keepdims=True)
        return U

    def _metric_calculation(self, labels=None, m=2):
        if labels is None:
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
        return metrics

    def plot_clusters_2d(self, labels=None, title="Cluster plot (2D PCA)"):
        if labels is None:
            labels = self.labels_

        X = self.data
        if X.shape[1] > 2:
            X2 = PCA(n_components=2, random_state=self.random_state).fit_transform(X)
            x0, x1 = X2[:, 0], X2[:, 1]
            #xlabel = "PC1"
            #ylabel = "PC2"
        else:
            x0, x1 = X[:, 0], X[:, 1]
            #xlabel = "Feature 0"
            #ylabel = "Feature 1"

        plt.figure()
        plt.scatter(x0, x1, c=labels)
        #lt.xlabel(xlabel)
        #plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()


def main():
    iris = load_iris()
    X = iris.data

    ca = ClasterAnalysis(X, n_clusters=3, method="nearest_neighbour")
    model = ca._clustering_procedure()
    print('Labels (first 10) ', ca.labels_[:10])
    print('Inertia ', ca.inertia_)
    print('Model ', model)

    print('Hopkins ', ca.hopkins_statistic_calculation(sampling_size=40))
    ks, sils = ca.silhouette_method(k_min=2, k_max=6)
    print('k_opt (silhouette) ', ca.k_opt)
    print('ks, sils ',ks, sils)

    gap_res = ca.gap_statistic(k_min=1, k_max=6, B=20)
    print('k_opt (gap) ', ca.k_opt)
    print('gap_res ',gap_res)

    metrics = ca._metric_calculation()
    print('Metrics ', metrics)

    ca.plot_clusters_2d()

main()