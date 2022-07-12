"""
Library for building clusters on data.
"""
from typing import List

from sklearn import cluster, preprocessing

import matplotlib.pyplot as plt

import plot_data
from plot_data.core import Dataset
import dessia_common.core as dc


class ClusterResult(dc.DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ['from_agglomerative_clustering',
                        'from_kmeans', 'from_dbscan']

    def __init__(self, data: dc.HeterogeneousList = None, labels: List[int] = None, name: str = ''):
        """
        Cluster object to instantiate and compute clusters on data.

        :param data: The future clustered data, defaults to None
        :type data: List[dc.DessiaObject], optional

        :param labels: The list of data labels, ordered the same as data, defaults to None
        :type labels: List[int], optional

        :param name: The name of ClusterResult object, defaults to ''
        :type name: str, optional

        """
        dc.DessiaObject.__init__(self, name=name)
        self.data = data
        self.labels = labels
        self.n_clusters = self.set_n_clusters()

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
        skl_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,
                                                      distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster = cls.fit_cluster(skl_cluster, data, scaling)
        return cls(data, skl_cluster.labels_.tolist())

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
        skl_cluster = cluster.KMeans(
            n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster = cls.fit_cluster(skl_cluster, data, scaling)
        return cls(data, skl_cluster.labels_.tolist())

    @classmethod
    def from_dbscan(cls, data: dc.HeterogeneousList, eps: float = 0.5, min_samples: int = 5,
                    mink_power: float = 2, leaf_size: int = 30, scaling: bool = False):
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

        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        This is the most important DBSCAN parameter to choose appropriately for your data
        set and distance function, defaults to 0.5
        :type eps: float, optional

        :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself, defaults to 5
        :type min_samples: int, optional

        :param mink_power: The power of the Minkowski metric to be used to calculate distance between points.
        If None, then mink_power=2 (equivalent to the Euclidean distance), defaults to 2
        :type mink_power: float, optional

        :param leaf_size: Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query,
        as well as the memory required to store the tree. The optimal value depends on the nature of the problem, defaults to 30
        :type leaf_size: int, optional

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
        skl_cluster = cluster.DBSCAN(
            eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size)
        skl_cluster = cls.fit_cluster(skl_cluster, data, scaling)
        return cls(data, skl_cluster.labels_.tolist())

    @staticmethod
    def fit_cluster(skl_cluster: cluster, data: dc.HeterogeneousList, scaling: bool = False):
        if scaling:
            scaled_matrix = ClusterResult.scale_data(data.matrix)
        else:
            scaled_matrix = data.matrix
        skl_cluster.fit(scaled_matrix)
        return skl_cluster

    @staticmethod
    def scale_data(data_matrix: List[List[float]]):
        scaled_matrix = preprocessing.StandardScaler().fit_transform(data_matrix)
        return list([list(map(float, row)) for row in scaled_matrix])

    def set_n_clusters(self):
        if self.labels is None:
            n_clusters = 0
        else:
            n_clusters = max(self.labels) + 1
        return n_clusters


# Here because of cyclic import if in core.py
class CategorizedList(dc.HeterogeneousList):
    def __init__(self, dessia_objects: List[dc.DessiaObject], labels: List[int], name: str = ''):
        dc.HeterogeneousList.__init__(self, dessia_objects=dessia_objects, name=name)
        if name == '':
            self.name += "unnamed_CategorizedList"
        self.labels = labels

    @property
    def n_clusters(self):
        return max(self.labels) + 1

    def clustered_sublists(self):
        sublists = []
        label_tags = sorted(list(map(str, set(self.labels).difference({-1}))))
        for i in range(max(self.labels) + 1):
            sublists.append([])
        if -1 in self.labels:
            sublists.append([])
            label_tags.append("outliers")

        for i, label in enumerate(self.labels):
            sublists[label].append(self.dessia_objects[i])

        return [dc.HeterogeneousList(dessia_objects=sublist, name=self.name + f"_{label_tag}")
                for label_tag, sublist in zip(label_tags, sublists)]

    def plot_data(self):
        # Plot a correlation matrix when plot_data.heatmap will be improved
        # correlation_matrix = []

        # Dimensionality plot
        dimensionality_plot = self.plot_dimensionality()

        # Scattermatrix
        datasets_list = self.build_datasets()
        scatter_matrix = self.build_multiplot(datasets_list)

        return [dimensionality_plot, scatter_matrix]

    def build_multiplot(self, datasets_list: List[Dataset]):
        list_scatters = []
        for x_num in range(len(self.common_attributes)):
            for y_num in range(len(self.common_attributes)):
                list_scatters.append(plot_data.Graph2D(x_variable=self.common_attributes[x_num],
                                                       y_variable=self.common_attributes[y_num],
                                                       graphs=datasets_list))
        return plot_data.MultiplePlots(plots=list_scatters, elements=datasets_list,
                                       initial_view_on=True)

    def build_datasets(self):
        dataset_list = []
        tooltip_list = []
        nb_dataset = (self.n_clusters if -1 not in self.labels else self.n_clusters + 1)

        for i_label in range(nb_dataset):
            dataset_list.append([])
            tooltip_list.append(plot_data.Tooltip(attributes=self.common_attributes + ["Cluster Label"]))

        for idx, label in enumerate(self.labels):
            dataset_row = {"Cluster Label": (
                label if label != -1 else "Excluded")}

            if hasattr(self.dessia_objects[idx], self.common_attributes[0]):
                for attribute in self.common_attributes:
                    dataset_row[attribute] = getattr(self.dessia_objects[idx], attribute)
            else:
                for col in range(len(self.matrix[0])):
                    dataset_row[f'p{col+1}'] = self.matrix[idx][col]

            dataset_list[label].append(dataset_row)

        cmp_f = plt.cm.get_cmap(
            'hsv', self.n_clusters + 1)(range(self.n_clusters + 1))
        edge_style = plot_data.EdgeStyle(line_width=0.0001)
        for idx in range(nb_dataset):
            if idx == self.n_clusters:
                color = plot_data.colors.Color(0, 0, 0)
            else:
                color = plot_data.colors.Color(
                    cmp_f[idx][0], cmp_f[idx][1], cmp_f[idx][2])
            point_style = plot_data.PointStyle(
                color_fill=color, color_stroke=color, size=1)
            dataset_list[idx] = plot_data.Dataset(elements=dataset_list[idx],
                                                  edge_style=edge_style,
                                                  point_style=point_style,
                                                  tooltip=tooltip_list[idx])
        return dataset_list




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
