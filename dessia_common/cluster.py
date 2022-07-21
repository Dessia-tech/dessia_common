"""
Library for building clusters on data.
"""
from typing import List

import numpy as npy
from sklearn import cluster, manifold

import matplotlib.pyplot as plt

import plot_data
import dessia_common.core as dc


class ClusterResult(dc.DessiaObject):
    """
    Cluster object to instantiate and compute clusters on data.
    """
    _standalone_in_db = True
    _allowed_methods = ['from_agglomerative_clustering', 'from_kmeans', 'from_dbscan']

    def __init__(self, data: List[dc.DessiaObject] = None, 
                 labels: List[int] = None, name: str = ''):
        dc.DessiaObject.__init__(self, name=name)
        self.data = data
        self.labels = labels
        self.n_clusters = self.set_n_clusters()

    @classmethod
    def from_agglomerative_clustering(cls, data: List[dc.DessiaObject], n_clusters: int = 2,
                                      affinity: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None):
        """
        Internet doc
        ----------
            Hierarchical clustering is a general family of clustering algorithms that
            build nested clusters by merging or splitting them successively.
            This hierarchy of clusters is represented as a tree (or dendrogram).
            The root of the tree is the unique cluster that gathers all the samples,
            the leaves being the clusters with only one sample. See the Wikipedia page
            for more details.

            The AgglomerativeClustering object performs a hierarchical clustering using
            a bottom up approach: each observation starts in its own cluster, and clusters
            are successively merged together. The linkage criteria determines the metric
            used for the merge strategy: Ward minimizes the sum of squared differences within all clusters.

            It is a variance-minimizing approach and in this sense is similar to the
            k-means objective function but tackled with an agglomerative hierarchical approach.
            Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
            Average linkage minimizes the average of the distances between all observations of pairs of clusters.
            Single linkage minimizes the distance between the closest observations of pairs of clusters.
            AgglomerativeClustering can also scale to large number of samples when it is used
            jointly with a connectivity matrix, but is computationally expensive when no connectivity
            constraints are added between samples: it considers at each step all the possible merges.

            See more : https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

        Parameters
        ----------
        cls : ClusterResult class object

        data : List[dc.DessiaObject]
            list of DessiaObject.

        n_clusters : int or None, default = 2
            The number of clusters to find.
            It must be None if distance_threshold is not None.

        affinity : str or callable, default = ’euclidean’
            Metric used to compute the linkage.
            Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
            If linkage is “ward”, only “euclidean” is accepted.
            If “precomputed”, a distance matrix (instead of a similarity matrix)
            is needed as input for the fit method.

        linkage: {‘ward’, ‘complete’, ‘average’, ‘single’}, default = ’ward’
            Which linkage criterion to use.
            The linkage criterion determines which distance to use between sets of observation.
            The algorithm will merge the pairs of cluster that minimize this criterion.
            - ‘ward’ minimizes the variance of the clusters being merged.
            - ‘average’ uses the average of the distances of each observation of the two sets.
            - ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
            - ‘single’ uses the minimum of the distances between all observations of the two sets.

        distance_threshold : float, default = None
            The linkage distance threshold above which, clusters will not be merged.
            If not None, n_clusters must be None and compute_full_tree must be True.

        """
        skl_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,
                                                      distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())

    @classmethod
    def from_kmeans(cls, data: List[dc.DessiaObject], n_clusters: int = 2,
                    n_init: int = 10, tol: float = 1e-4):
        """
        Internet doc
        ----------
            The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
            minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below).
            This algorithm requires the number of clusters to be specified. It scales well to large number
            of samples and has been used across a large range of application areas in many different fields.
            The k-means algorithm divides a set of samples into disjoint clusters , each described by the mean
            of the samples in the cluster. The means are commonly called the cluster “centroids”; note that
            they are not, in general, points from, although they live in the same space.
            The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster
            sum-of-squares criterion.

            See more : https://scikit-learn.org/stable/modules/clustering.html#k-means

        Parameters
        ----------
        cls : ClusterResult class object

        data : List[dc.DessiaObject]
            list of DessiaObject.

        n_clusters : int, default = 8
            The number of clusters to form as well as the number of centroids to generate.

        n_init : int, default = 10
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.

        tol : float, default = 1e-4
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
            of two consecutive iterations to declare convergence.

        """
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())

    @classmethod
    def from_dbscan(cls, data: List[dc.DessiaObject], eps: float = 0.5, min_samples: int = 5,
                    mink_power: float = 2, leaf_size: int = 30):
        """
        Internet doc
        ----------
            The DBSCAN algorithm views clusters as areas of high density separated by areas of low density.
            Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means
            which assumes that clusters are convex shaped. The central component to the DBSCAN is the concept
            of core samples, which are samples that are in areas of high density. A cluster is therefore a set
            of core samples, each close to each other (measured by some distance measure) and a set of non-core
            samples that are close to a core sample (but are not themselves core samples).
            There are two parameters to the algorithm, min_samples and eps, which define formally what we mean
            when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.

            See more : https://scikit-learn.org/stable/modules/clustering.html#dbscan

        Parameters
        ----------
        cls : ClusterResult class object

        data : List[dc.DessiaObject]
            list of DessiaObject.

        eps : float, default = 0.5
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data
            set and distance function.

        min_samples : int, default = 5
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.

        mink_power : float, default = None
            The power of the Minkowski metric to be used to calculate distance between points.
            If None, then p=2 (equivalent to the Euclidean distance).

        leaf_size : int, default = 30
            Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query,
            as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

        """
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size)
        skl_cluster.fit(cls.to_matrix(data))
        skl_cluster.labels_ += 1  # To test
        return cls(data, skl_cluster.labels_.tolist())

    @staticmethod
    def to_matrix(data: List[dc.DessiaObject]):
        if 'to_vector' not in dir(data[0]):
            raise NotImplementedError(f"{data[0].__class__.__name__} objects must have a " +
                                      "'to_vector' method to be handled in ClusterResult object.")
        data_matrix = []
        for element in data:
            data_matrix.append(element.to_vector())
        return data_matrix

    @staticmethod
    # Is it really pertinent to have a staticmethod for that since we will only call it when having a ClusterResult
    def data_to_clusters(data: List[dc.DessiaObject], labels: npy.ndarray):
        clusters_list = []
        for i in range(npy.max(labels) + 1):
            clusters_list.append([])
        for i, label in enumerate(labels):
            clusters_list[label].append(data[i])
        return clusters_list

    def set_n_clusters(self):
        if self.labels is None:
            n_clusters = 0
        else:
            n_clusters = max(self.labels) + 1
        return n_clusters

    def check_dimensionality(self):
        _, singular_values, _ = npy.linalg.svd(self.to_matrix(self.data))
        normed_singular_values = singular_values / npy.sum(singular_values)
        plt.figure()
        plt.semilogy(normed_singular_values, linestyle='None', marker='o')
        plt.grid()
        plt.title("Normalized singular values of data")
        plt.xlabel("Index of reduced basis vector")
        plt.ylabel("Singular value")

    def plot_data(self):
        n_clusters = npy.max(self.labels) + 1
        encoding_mds = manifold.MDS(metric=True, n_jobs=-1, n_components=2)
        matrix_mds = encoding_mds.fit_transform(self.to_matrix(self.data))

        elements = []
        for i in range(len(matrix_mds)):
            elements.append({"X_MDS": matrix_mds[i, 0].tolist(),
                             "Y_MDS": matrix_mds[i, 1]})

        dataset_list = []
        for i in range(n_clusters):
            dataset_list.append([])
        for i, label in enumerate(self.labels):
            dataset_list[label].append({"X_MDS": matrix_mds[i, 0].tolist(),
                                        "Y_MDS": matrix_mds[i, 1]})

        cmp_f = plt.cm.get_cmap('jet', n_clusters)(range(n_clusters))
        edge_style = plot_data.EdgeStyle(line_width=0.0001)
        for i in range(n_clusters):
            color = plot_data.colors.Color(cmp_f[i][0], cmp_f[i][1], cmp_f[i][2])
            point_style = plot_data.PointStyle(color_fill=color, color_stroke=color)
            dataset_list[i] = plot_data.Dataset(elements=dataset_list[i],
                                                edge_style=edge_style,
                                                point_style=point_style)

        scatter_plot = plot_data.Graph2D(x_variable="X_MDS",
                                         y_variable="Y_MDS",
                                         graphs=dataset_list)

        return plot_data.plot_canvas(plot_data_object=scatter_plot, debug_mode=True)
