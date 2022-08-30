"""
Library for building clusters on data.
"""
import itertools
from typing import List

import numpy as npy
from sklearn import cluster
import matplotlib.pyplot as plt

try:
    import plot_data
except ImportError:
    pass
import dessia_common.core as dc

class CategorizedList(dc.HeterogeneousList):
    _allowed_methods = ['from_agglomerative_clustering', 'from_kmeans', 'from_dbscan']


    def __init__(self, dessia_objects: List[dc.DessiaObject] = None, labels: List[int] = None, name: str = ''):
        dc.HeterogeneousList.__init__(self, dessia_objects=dessia_objects, name=name)
        self.labels = labels
        self._n_clusters = None

    @property
    def n_clusters(self):
        if self._n_clusters is None:
            self._n_clusters = max(self.labels) + 1
        return self._n_clusters

    def clustered_sublists(self):
        sublists = []
        label_tags = sorted(list(map(str, set(self.labels).difference({-1}))))
        for _ in range(max(self.labels) + 1):
            sublists.append([])
        if -1 in self.labels:
            sublists.append([])
            label_tags.append("outliers")

        for idx, label in enumerate(self.labels):
            sublists[label].append(self.dessia_objects[idx])

        new_dessia_objects = [dc.HeterogeneousList(dessia_objects=sublist, name=self.name + f"_{label_tag}")
                              for label_tag, sublist in zip(label_tags, sublists)]

        return CategorizedList(new_dessia_objects,
                               list(set(self.labels).difference({-1})) + ([-1] if -1 in self.labels else []),
                               name=self.name + "_split")

    def _tooltip_attributes(self):
        return self.common_attributes + ["Cluster Label"]

    def _plot_data_list(self):
        _plot_data_list = []
        for row, label in enumerate(self.labels):
            _plot_data_list.append({attr: self.matrix[row][col] for col, attr in enumerate(self.common_attributes)})
            _plot_data_list[-1]["Cluster Label"] = label
            # (label if label != -1 else "Excluded") plot_data "Excluded" -> NaN
        return _plot_data_list


    def _point_families(self):
        colormap = plt.cm.get_cmap('hsv', self.n_clusters + 1)(range(self.n_clusters + 1))
        point_families = []
        for i_cluster in range(self.n_clusters):
            color = plot_data.colors.Color(colormap[i_cluster][0], colormap[i_cluster][1], colormap[i_cluster][2])
            points_index = list(map(int, npy.where(npy.array(self.labels) == i_cluster)[0].tolist()))
            point_families.append(plot_data.core.PointFamily(color, points_index))

        if -1 in self.labels:
            color = plot_data.colors.LIGHTGREY
            points_index =  list(map(int, npy.where(npy.array(self.labels) == -1)[0].tolist()))
            point_families.append(plot_data.core.PointFamily(color, points_index))
        return point_families


    @classmethod
    def from_agglomerative_clustering(cls, data: dc.HeterogeneousList, n_clusters: int = 2,
                                      affinity: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None, scaling: bool = False):

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

        :param data: The future clustered data.
        :type data: List[dc.DessiaObject]

        :param n_clusters: number of wished clusters, defaults to 2,
            Must be None if distance_threshold is not None
        :type n_clusters: int, optional

        :param affinity: metric used to compute the linkage, defaults to 'euclidean'.
            Can be one of [“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”].
            If linkage is “ward”, only “euclidean” is accepted.
            If “precomputed”, a distance matrix (instead of a similarity matrix)
            is needed as input for the fit method.
        :type affinity: str, optional

        :param linkage: Which linkage criterion to use, defaults to 'ward'
            Can be one of [‘ward’, ‘complete’, ‘average’, ‘single’]
            The linkage criterion determines which distance to use between sets of observation.
            The algorithm will merge the pairs of cluster that minimize this criterion.
                - ‘ward’ minimizes the variance of the clusters being merged.
                - ‘average’ uses the average of the distances of each observation of the two sets.
                - ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
                - ‘single’ uses the minimum of the distances between all observations of the two sets.
        :type linkage: str, optional

        :param distance_threshold: The linkage distance above which clusters will not be merged, defaults to None
            If not None, n_clusters must be None.
        :type distance_threshold: float, optional

        :param scaling: Whether to scale the data or not before clustering.
        Formula is scaled_x = ( x - mean ) / standard_deviation, default to False
        :type scaling: bool, optional

        :return: a ClusterResult object that knows the data and their labels
        :rtype: ClusterResult

        """
        skl_cluster = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist())

    @classmethod
    def from_kmeans(cls, data: dc.HeterogeneousList, n_clusters: int = 2,
                    n_init: int = 10, tol: float = 1e-4, scaling: bool = False):

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

        :param data: The future clustered data.
        :type data: List[dc.DessiaObject]

        :param n_clusters: number of wished clusters, defaults to 2
        :type n_clusters: int, optional

        :param n_init: Number of time the k-means algorithm will be run with different centroid seeds, defaults to 10
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        :type n_init: int, optional

        :param tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
            of two consecutive iterations to declare convergence., defaults to 1e-4
        :type tol: float, optional

        :param scaling: Whether to scale the data or not before clustering.
        Formula is scaled_x = ( x - mean ) / standard_deviation, default to False
        :type scaling: bool, optional

        :return: a ClusterResult object that knows the data and their labels
        :rtype: ClusterResult

        """
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist())

    @classmethod
    def from_dbscan(cls, data: dc.HeterogeneousList, eps: float = 0.5, min_samples: int = 5, mink_power: float = 2,
                    leaf_size: int = 30, metric: str = "euclidean", scaling: bool = False):

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

        :param data: The future clustered data.
        :type data: List[dc.DessiaObject]

        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood
        of the other. This is not a maximum bound on the distances of points within a cluster.
        This is the most important DBSCAN parameter to choose appropriately for your data
        set and distance function, defaults to 0.5
        :type eps: float, optional

        :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as
        a core point. This includes the point itself, defaults to 5
        :type min_samples: int, optional

        :param mink_power: The power of the Minkowski metric to be used to calculate distance between points.
        If None, then mink_power=2 (equivalent to the Euclidean distance), defaults to 2
        :type mink_power: float, optional

        :param leaf_size: Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction
        and query, as well as the memory required to store the tree. The optimal value depends on the nature of
        the problem, defaults to 30
        :type leaf_size: int, optional

        :param metric: The metric to use when calculating distance between instances in a feature array.
        If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances
        for its metric parameter. If metric is “precomputed”, X is assumed to be a distance matrix and must be square.
        X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors for DBSCAN.
        :type metric: str, or callable, default=’euclidean’


        :param scaling: Whether to scale the data or not before clustering.
        Formula is scaled_x = ( x - mean ) / standard_deviation, default to False
        :type scaling: bool, optional

        :return: a ClusterResult object that knows the data and their labels
        :rtype: ClusterResult

        !! WARNING !!
        ----------
            All labels are summed with 1 in order to improve the code simplicity and ease to use.
            Then -1 labelled values are now at 0 and must not be considered as clustered values when using DBSCAN.

        """
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size, metric=metric)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist())

    @classmethod
    def from_pareto_sheets(cls, h_list: dc.HeterogeneousList, costs: List[List[float]], nb_sheets: int = 1):
        labels = []
        dessia_objects = []
        # TODO: __getitem__
        pareto_sheets, non_optimal_points = h_list.pareto_sheets(costs, nb_sheets)
        for label, pareto_sheet in enumerate(pareto_sheets):
            labels.extend([label]*len(pareto_sheet.dessia_objects))
            dessia_objects.extend(pareto_sheet.dessia_objects)
        dessia_objects.extend(non_optimal_points.dessia_objects)
        labels.extend([label + 1]*len(non_optimal_points.dessia_objects))
        return cls(dessia_objects, labels)

    @staticmethod
    def fit_cluster(skl_cluster: cluster, matrix: List[List[float]], scaling: bool):
        if scaling:
            scaled_matrix = dc.HeterogeneousList.scale_data(matrix)
        else:
            scaled_matrix = matrix
        skl_cluster.fit(scaled_matrix)
        return skl_cluster

# Function to implement, to find a good eps parameter for dbscan
# def nearestneighbors(self):
#     vectors = []
#     for machine in self.machines:
#         vector = machine.to_vector()
#         vectors.append(vector)
#     neigh = NearestNeighbors(n_neighbors=14)
#     vectors = StandardScaler().fit_transform(vectors)
#     nbrs = neigh.fit(vectors)
#     distances, indices = nbrs.kneighbors(vectors)
#     distances = npy.sort(distances, axis=0)
#     distances = distances[:, 1]
#     plt.plot(distances)
#     plt.show()
