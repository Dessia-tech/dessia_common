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
    """
    Base object for handling a categorized (clustered) list of DessiaObjects.

    **CategorizedList should be instantiated with** `from_...` **methods.**

    **Do not use** `__init__` **to instantiate a CategorizedList.**

    :param dessia_objects:
        --------
        List of DessiaObjects to store in CategorizedList
    :type dessia_objects: `List[DessiaObject]`, `optional`, defaults to `None`

    :param labels:
        --------
        Labels of DessiaObjects' cluster stored in CategorizedList
    :type labels: `List[int]`, `optional`, defaults to `None`

    :param name:
        --------
        Name of CategorizedList
    :type name: `str`, `optional`, defaults to `""`

    :Properties:
        * **common_attributes:** (`List[str]`)
            --------
            Common attributes of DessiaObjects contained in the current `CategorizedList`
        * **matrix:** (`List[List[float]]`, `n_samples x n_features`)
            --------
            Matrix of data computed by calling the to_vector method of all dessia_objects
        * **n_cluster:** (`int`)
            --------
            Number of clusters in dessia_objects

    **Built-in methods**: See :func:`~dessia_common.core.HeterogeneousList`
    """
    _allowed_methods = ['from_agglomerative_clustering', 'from_kmeans', 'from_dbscan']

    def __init__(self, dessia_objects: List[dc.DessiaObject] = None, labels: List[int] = None, name: str = ''):
        dc.HeterogeneousList.__init__(self, dessia_objects=dessia_objects, name=name)
        if labels is None:
            labels = [0]*len(self.dessia_objects)
        self.labels = labels
        self._n_clusters = None

    @property
    def n_clusters(self):
        if self._n_clusters is None:
            unic_labels = set(self.labels)
            unic_labels.discard(-1)
            self._n_clusters = len(unic_labels)
        return self._n_clusters

    def pick_from_slice(self, key: slice):
        new_hlist = dc.HeterogeneousList.pick_from_slice(self, key)
        new_hlist.labels = self.labels[key]
        # new_hlist.name += f"_{key.start if key.start is not None else 0}_{key.stop}")
        return new_hlist

    def pick_from_boolist(self, key: List[bool]):
        new_hlist = dc.HeterogeneousList.pick_from_boolist(self, key)
        new_hlist.labels = dc.DessiaFilter.apply(self.labels, key)
        # new_hlist.name += "_list")
        return new_hlist

    def __str__(self):
        label_space = 4
        print_lim = 15
        attr_name_len = []
        attr_space = []
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += (f"{len(self.dessia_objects)} samples, {len(self.common_attributes)} features, {self.n_clusters} " +
                   "clusters")
        if self.__len__() == 0:
            return prefix

        string = "|" + " "*(label_space - 1) + "n°" + " "*(label_space - 1)
        string += self._print_titles(attr_space, attr_name_len)

        string += "\n" + "-"*len(string)
        for label, dessia_object in zip(self.labels, self.dessia_objects[:print_lim]):
            string += "\n"
            space = 2*label_space - 1 - len(str(label))
            string += "|" + " "*space + f"{label}" + " "

            string += self._print_objects(dessia_object, attr_space, attr_name_len)

        return prefix + "\n" + string + "\n"

    def clustered_sublists(self):
        """
        Split a CategorizedList of labelled DessiaObjects into a CategorizedList of labelled HeterogeneousLists.

        :return: A CategorizedList of length n_cluster that store each cluster in a HeterogeneousList. Labels are \
            the labels of each cluster, i.e. stored HeterogeneousList
        :rtype: CategorizedList[HeterogeneousList]

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = HeterogeneousList(all_cars_wi_feat, name="cars")
        >>> clist = CategorizedList.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> split_clist = clist.clustered_sublists()
        >>> print(split_clist[:3])
        CategorizedList ex_split: 3 samples, 2 features, 3 clusters
        |   n°   |   Name   |   Common_attributes   |
        ---------------------------------------------
        |      0 |     ex_0 |['mpg', 'displacemen...|
        |      1 |     ex_1 |['mpg', 'displacemen...|
        |      2 |     ex_2 |['mpg', 'displacemen...|
        >>> print(split_clist[3][:3])
        HeterogeneousList ex_3: 3 samples, 5 features
        |   Mpg   |   Displacement   |   Horsepower   |   Weight   |   Acceleration   |
        -------------------------------------------------------------------------------
        |    21.0 |              0.2 |           85.0 |     2587.0 |             16.0 |
        |    25.0 |             0.11 |           87.0 |     2672.0 |             17.5 |
        |    21.0 |            0.199 |           90.0 |     2648.0 |             15.0 |
        """
        sublists = []
        label_tags = sorted(list(map(str, set(self.labels).difference({-1}))))
        unic_labels = list(set(self.labels))
        for _ in range(self.n_clusters):
            sublists.append([])
        if -1 in self.labels:
            sublists.append([])
            label_tags.append("outliers")

        for idx, label in enumerate(self.labels):
            sublists[unic_labels.index(label)].append(self.dessia_objects[idx])

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
            point_families.append(plot_data.core.PointFamily(color, points_index, name="Cluster " + str(i_cluster)))

        if -1 in self.labels:
            color = plot_data.colors.LIGHTGREY
            points_index = list(map(int, npy.where(npy.array(self.labels) == -1)[0].tolist()))
            point_families.append(plot_data.core.PointFamily(color, points_index, name="Excluded"))
        return point_families

    @classmethod
    def from_agglomerative_clustering(cls, data: dc.HeterogeneousList, n_clusters: int = 2,
                                      affinity: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None, scaling: bool = False, name: str =""):

        """
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
        :type data: List[DessiaObject]

        :param n_clusters:
            -------
            Number of wished clusters.

            Must be `None` if `distance_threshold` is not `None`
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param affinity:
            -------
            Metric used to compute the linkage.
            Can be one of `['euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed']`.

            If linkage is `'ward'`, only `'euclidean'` is accepted.

            If `'precomputed'`, a distance matrix (instead of a similarity matrix) is needed as input for the \
                fit method.
        :type affinity: `str`, `optional`, defaults to `'euclidean'`

        :param linkage:
            --------
            |  Which linkage criterion to use. Can be one of `[‘ward’, ‘complete’, ‘average’, ‘single’]`
            |  The linkage criterion determines which distance to use between sets of observation.
            |  The algorithm will merge the pairs of cluster that minimize this criterion.
            |     - `'ward'` minimizes the variance of the clusters being merged.
            |     - `'average`' uses the average of the distances of each observation of the two sets.
            |     - `'complete`' or `'maximum`' linkage uses the maximum distances between all observations of the two \
                sets.
            |     - `'single`' uses the minimum of the distances between all observations of the two sets.
        :type linkage: `str`, `optional`, defaults to `'ward'`

        :param distance_threshold:
            --------
            The linkage distance above which clusters will not be merged.
            If not `None`, `n_clusters` must be `None`.
        :type distance_threshold: `float`, `optional`, defaults to `None`

        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedListt
        """
        skl_cluster = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_kmeans(cls, data: dc.HeterogeneousList, n_clusters: int = 2, n_init: int = 10, tol: float = 1e-4,
                    scaling: bool = False, name: str =""):
        """
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
        :type data: List[DessiaObject]

        :param n_clusters:
            --------
            Number of wished clusters
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param n_init:
            --------
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        :type n_init: `int`, `optional`, defaults to `10`

        :param tol:
            --------
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two \
                consecutive iterations to declare convergence.
        :type tol: `float`, `optional`, defaults to `1e-4`

        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedList
        """
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_dbscan(cls, data: dc.HeterogeneousList, eps: float = 0.5, min_samples: int = 5, mink_power: float = 2,
                    leaf_size: int = 30, metric: str = "euclidean", scaling: bool = False, name: str =""):

        """
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
        :type data: List[DessiaObject]

        :param eps:
            --------
            The maximum distance between two samples for one to be considered as in the neighborhood \
            of the other. This is not a maximum bound on the distances of points within a cluster. This is the most \
            important DBSCAN parameter to choose appropriately for your data set and distance function
        :type eps: `float`, `optional`, defaults to `0.5`

        :param min_samples:
            --------
            The number of samples (or total weight) in a neighborhood for a point to be considered as \
            a core point. This includes the point itself
        :type min_samples: `int`, `optional`, defaults to 5

        :param mink_power:
            --------
            The power of the Minkowski metric to be used to calculate distance between points. \
            If `None`, then `mink_power=2` (equivalent to the Euclidean distance)
        :type mink_power: `float`, `optional`, defaults to `2`

        :param leaf_size:
            --------
            Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and \
            query, as well as the memory required to store the tree. The optimal value depends on the nature of the \
            problem
        :type leaf_size: `int`, `optional`, defaults to `30`

        :param metric:
            --------
            The metric to use when calculating distance between instances in a feature array. If metric is \
            a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its \
            metric parameter. If metric is `'precomputed'`, X is assumed to be a distance matrix and must be square. \
            X may be a sparse graph, in which case only `'nonzero'` elements may be considered neighbors for DBSCAN.
        :type metric: `str`, or `callable`, default to `’euclidean’`


        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedList
        """
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size, metric=metric)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_pareto_sheets(cls, h_list: dc.HeterogeneousList, costs: List[List[float]], nb_sheets: int = 1):
        labels = []
        dessia_objects = []
        pareto_sheets, non_optimal_points = h_list.pareto_sheets(costs, nb_sheets)
        for label, pareto_sheet in enumerate(pareto_sheets):
            labels.extend([label]*len(pareto_sheet))
            dessia_objects.extend(pareto_sheet)
        dessia_objects.extend(non_optimal_points)
        labels.extend([label + 1]*len(non_optimal_points))
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
