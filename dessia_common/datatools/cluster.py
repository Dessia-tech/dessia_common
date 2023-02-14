""" Library for building clusters on Dataset or List. """
from typing import List

from scipy.spatial.distance import cdist
import numpy as npy
from sklearn import cluster
import matplotlib.pyplot as plt

try:
    from plot_data.core import PointFamily
    from plot_data.colors import LIGHTGREY, Color
except ImportError:
    pass
from dessia_common.exports import XLSXWriter
from dessia_common.core import DessiaObject, DessiaFilter
from dessia_common.datatools.dataset import Dataset


class ClusteredDataset(Dataset):
    """
    Base object for handling a categorized (clustered) list of DessiaObjects.

    **ClusteredDataset should be instantiated with** `from_...` **methods.**

    **Do not use** `__init__` **to instantiate a ClusteredDataset.**

    :param dessia_objects:
        --------
        List of DessiaObjects to store in ClusteredDataset
    :type dessia_objects: `List[DessiaObject]`, `optional`, defaults to `None`

    :param labels:
        --------
        Labels of DessiaObjects' cluster stored in ClusteredDataset
    :type labels: `List[int]`, `optional`, defaults to `None`

    :param name:
        --------
        Name of ClusteredDataset
    :type name: `str`, `optional`, defaults to `""`

    :Properties:
        * **common_attributes:** (`List[str]`)
            --------
            Common attributes of DessiaObjects contained in the current `ClusteredDataset`
        * **matrix:** (`List[List[float]]`, `n_samples x n_features`)
            --------
            Matrix of data computed by calling the to_vector method of all dessia_objects
        * **n_cluster:** (`int`)
            --------
            Number of clusters in dessia_objects

    **Built-in methods**: See :func:`~Dataset`
    """

    _allowed_methods = ['from_agglomerative_clustering', 'from_kmeans', 'from_dbscan', 'from_pareto_sheets']

    def __init__(self, dessia_objects: List[DessiaObject] = None, labels: List[int] = None, name: str = ''):
        """ See class docstring. """
        Dataset.__init__(self, dessia_objects=dessia_objects, name=name)
        if labels is None:
            labels = [0] * len(self)
        self.labels = labels

    @property
    def n_clusters(self):
        """ Number of clusters in dessia_objects. """
        unic_labels = set(self.labels)
        unic_labels.discard(-1)
        return len(unic_labels)

    def to_xlsx_stream(self, stream):
        """ Export the object to an XLSX to a given stream. """
        if not isinstance(self.dessia_objects[0], Dataset):
            writer = XLSXWriter(self.clustered_sublists())
        else:
            writer = XLSXWriter(self)
        writer.save_to_stream(stream)

    def _pick_from_slice(self, key: slice):
        new_hlist = Dataset._pick_from_slice(self, key)
        new_hlist.labels = self.labels[key]
        # new_hlist.name += f"_{key.start if key.start is not None else 0}_{key.stop}")
        return new_hlist

    def _pick_from_boolist(self, key: List[bool]):
        new_hlist = Dataset._pick_from_boolist(self, key)
        new_hlist.labels = DessiaFilter.apply(self.labels, key)
        # new_hlist.name += "_list")
        return new_hlist

    def _printed_attributes(self):
        return ["label"] + Dataset._printed_attributes(self)

    def _write_str_prefix(self):
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += (f"{len(self)} samples, {len(self.common_attributes)} features, {self.n_clusters} clusters")
        return prefix

    def _get_printed_value(self, index: int, attr: str):
        if attr not in ["label"]:
            return super()._get_printed_value(index, attr)
        return self.labels[index]

    def clustered_sublists(self):
        """
        Split a ClusteredDataset of labelled DessiaObjects into aClusteredDatasetet of labelled Datasets.

        :return: A ClusteredDataset of length n_cluster that store each cluster in a Dataset. Labels are \
            the labels of each cluster, i.e. stored Dataset
        :rtype: ClusteredDataset[Dataset]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.datatools.cluster import ClusteredDataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = Dataset(all_cars_wi_feat, name="cars")
        >>> clist = ClusteredDataset.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> split_clist = clist.clustered_sublists()
        >>> print(split_clist[:3])
        ClusteredDataset ex_split: 3 samples, 2 features, 3 clusters
        |   n°   |   Name   |   Common_attributes   |
        ---------------------------------------------
        |      0 |     ex_0 |['mpg', 'displacemen...|
        |      1 |     ex_1 |['mpg', 'displacemen...|
        |      2 |     ex_2 |['mpg', 'displacemen...|
        >>> print(split_clist[3][:3])
        Dataset ex_3: 3 samples, 5 features
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

        new_dessia_objects = [Dataset(dessia_objects=sublist, name=self.name + f"_{label_tag}")
                              for label_tag, sublist in zip(label_tags, sublists)]

        return ClusteredDataset(new_dessia_objects,
                                list(set(self.labels).difference({-1})) + ([-1] if -1 in self.labels else []),
                                name=self.name + "_split")

    def _check_transform_sublists(self):
        if not isinstance(self.dessia_objects[0], Dataset):
            return self.clustered_sublists()
        return self[:]

    def mean_clusters(self):
        """
        Compute mathematical means of all clusters.

        Means are computed from the property `matrix`. Each element of the output is the average values in each
        dimension in one cluster.

        :return: A list of `n_cluster` lists of `n_samples` where each element is the average value in a dimension in \
        one cluster.
        :rtype: List[List[float]]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.datatools.cluster import ClusteredDataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = Dataset(all_cars_wi_feat, name="cars")
        >>> clist = ClusteredDataset.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> means = clist.mean_clusters()
        >>> print(means[0])
        [28.83333333333334, 0.10651785714285714, 79.16666666666667, 2250.3571428571427, 16.075000000000006]
        """
        clustered_sublists = self._check_transform_sublists()
        means = []
        for hlist in clustered_sublists:
            means.append(hlist.mean())
        return means

    def cluster_distances(self, method: str = 'minkowski', **kwargs):
        """
        Compute all distances between elements of each cluster and their mean.

        Gives an indicator on how clusters are built.

        :param method:
            --------
            Method to compute distances.
            Can be one of `[‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, \
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, \
            ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]`.
        :type method: `str`, `optional`, defaults to `'minkowski'`

        :param **kwargs:
            --------
            |  Extra arguments to metric: refer to each metric documentation for a list of all possible arguments.
            |  Some possible arguments:
            |     - p : scalar The p-norm to apply for Minkowski, weighted and unweighted. Default: `2`.
            |     - w : array_like The weight vector for metrics that support weights (e.g., Minkowski).
            |     - V : array_like The variance vector for standardized Euclidean. Default: \
                `var(vstack([XA, XB]), axis=0, ddof=1)`
            |     - VI : array_like The inverse of the covariance matrix for Mahalanobis. Default: \
                `inv(cov(vstack([XA, XB].T))).T`
            |     - out : ndarray The output array If not None, the distance matrix Y is stored in this array.
        :type **kwargs: `dict`, `optional`

        :return: `n_clusters` lists of distances of all elements of a cluster from its mean.
        :rtype: List[List[float]]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.datatools.cluster import ClusteredDataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = Dataset(all_cars_wi_feat, name="cars")
        >>> clist = ClusteredDataset.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> cluster_distances = clist.cluster_distances()
        >>> print(list(map(int, cluster_distances[6])))
        [180, 62, 162, 47, 347, 161, 160, 67, 164, 206, 114, 138, 97, 159, 124, 139]
        """
        clustered_sublists = self._check_transform_sublists()
        kwargs = self._set_distance_kwargs(method, kwargs)
        means = clustered_sublists.mean_clusters()
        cluster_distances = []
        for mean_, hlist in zip(means, clustered_sublists):
            cluster_distances.append(cdist([mean_], hlist.matrix, method, **kwargs).tolist()[0])
        return cluster_distances

    def cluster_real_centroids(self, method: str = 'minkowski', **kwargs):
        """
        In each cluster, finds the nearest existing element from the cluster's mean.

        :param method:
            --------
            Method to compute distances.
            Can be one of `[‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, \
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, \
            ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]`.
        :type method: `str`, `optional`, defaults to `'minkowski'`

        :param **kwargs:
            --------
            |  Extra arguments to metric: refer to each metric documentation for a list of all possible arguments.
            |  Some possible arguments:
            |     - p : scalar The p-norm to apply for Minkowski, weighted and unweighted. Default: `2`.
            |     - w : array_like The weight vector for metrics that support weights (e.g., Minkowski).
            |     - V : array_like The variance vector for standardized Euclidean. Default: \
               `var(vstack([XA, XB]), axis=0, ddof=1)`
            |     - VI : array_like The inverse of the covariance matrix for Mahalanobis. Default: \
               `inv(cov(vstack([XA, XB].T))).T`
            |     - out : ndarray The output array If not None, the distance matrix Y is stored in this array.
        :type **kwargs: `dict`, `optional`

        :return: `n_clusters` lists of distances of all elements of a cluster from its mean.
        :rtype: List[List[float]]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.datatools.cluster import ClusteredDataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = Dataset(all_cars_wi_feat, name="cars")
        >>> clist = ClusteredDataset.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> cluster_real_centroids = clist.cluster_real_centroids()
        >>> print(Dataset([cluster_real_centroids[0]]))
        Dataset 0x7f752654a0a0: 1 samples, 5 features
        |   Name   |   Mpg   |   Displacement   |   Horsepower   |   Weight   |   Acceleration   |
        ------------------------------------------------------------------------------------------
        |Dodge C...|    26.0 |            0.098 |           79.0 |     2255.0 |             17.7 |
        """
        clustered_sublists = self._check_transform_sublists()
        kwargs = self._set_distance_kwargs(method, kwargs)
        labels = clustered_sublists.labels
        cluster_distances = clustered_sublists.cluster_distances(method=method, **kwargs)
        real_centroids = [[] for _ in labels]
        for label in labels:
            min_idx = cluster_distances[label].index(min(cluster_distances[label]))
            real_centroids[label] = clustered_sublists.dessia_objects[label][min_idx]
        return real_centroids

    def _merge_sublists(self):
        merged_hlists = self.dessia_objects[0][:]
        merged_labels = [self.labels[0]] * len(merged_hlists)
        for dobject, label in zip(self.dessia_objects[1:], self.labels[1:]):
            merged_hlists.extend(dobject)
            merged_labels.extend([label] * len(dobject))
        plotted_clist = self.__class__(dessia_objects=merged_hlists.dessia_objects, labels=merged_labels)
        return plotted_clist

    def _tooltip_attributes(self):
        return self.common_attributes + ["Cluster Label"]

    def plot_data(self, reference_path: str = "#", **kwargs):
        """
        Plot data method.

        If dessia_objects are Dataset, merge all Dataset to plot them in one.
        """
        if isinstance(self.dessia_objects[0], Dataset):
            plotted_clist = self._merge_sublists()
            return plotted_clist.plot_data(reference_path=reference_path, **kwargs)
        return Dataset.plot_data(self, reference_path=reference_path, **kwargs)

    def _object_to_sample(self, dessia_object: DessiaObject, row: int, reference_path: str = '#'):
        sample = super()._object_to_sample(dessia_object=dessia_object, row=row, reference_path=reference_path)
        sample.values["Cluster Label"] = self.labels[row]
        return sample

    def _point_families(self):
        colormap = plt.cm.get_cmap('hsv', self.n_clusters + 1)(range(self.n_clusters + 1))
        point_families = []
        for i_cluster in range(self.n_clusters):
            color = Color(colormap[i_cluster][0], colormap[i_cluster][1], colormap[i_cluster][2])
            points_index = list(map(int, npy.where(npy.array(self.labels) == i_cluster)[0].tolist()))
            point_families.append(PointFamily(color, points_index, name="Cluster " + str(i_cluster)))

        if -1 in self.labels:
            color = LIGHTGREY
            points_index = list(map(int, npy.where(npy.array(self.labels) == -1)[0].tolist()))
            point_families.append(PointFamily(color, points_index, name="Excluded"))
        return point_families

    @classmethod
    def from_agglomerative_clustering(cls, data: Dataset, n_clusters: int = 2,
                                      metric: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None, scaling: bool = False, name: str = ""):
        """
        Agglomerative clustering on Dataset.

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
            Number of wished clusters.

            Must be `None` if `distance_threshold` is not `None`
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param metric:
            Metric used to compute the linkage.
            Can be one of `['euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed']`.

            If linkage is `'ward'`, only `'euclidean'` is accepted.

            If `'precomputed'`, a distance matrix (instead of a similarity matrix) is needed as input for the \
                fit method.
        :type metric: `str`, `optional`, defaults to `'euclidean'`

        :param linkage:
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
            The linkage distance above which clusters will not be merged.
            If not `None`, `n_clusters` must be `None`.
        :type distance_threshold: `float`, `optional`, defaults to `None`

        :param scaling:
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a ClusteredDataset that knows the data and their labels
        :rtype: ClusteredDataset
        """
        skl_cluster = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, metric=metric, distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_kmeans(cls, data: Dataset, n_clusters: int = 2, n_init: int = 10, tol: float = 1e-4,
                    scaling: bool = False, name: str = ""):
        """
        K-Means clustering on Dataset.

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
            Number of wished clusters
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param n_init:
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        :type n_init: `int`, `optional`, defaults to `10`

        :param tol:
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two \
                consecutive iterations to declare convergence.
        :type tol: `float`, `optional`, defaults to `1e-4`

        :param scaling:
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a ClusteredDataset that knows the data and their labels
        :rtype: ClusteredDataset
        """
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_dbscan(cls, data: Dataset, eps: float = 0.5, min_samples: int = 5, mink_power: float = 2,
                    leaf_size: int = 30, metric: str = "euclidean", scaling: bool = False, name: str = ""):
        """
        DBSCAN clustering on Dataset.

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
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN
            parameter to choose appropriately for your data set and distance function.
        :type eps: `float`, `optional`, defaults to `0.5`

        :param min_samples:
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.
        :type min_samples: `int`, `optional`, defaults to 5

        :param mink_power:
            The power of the Minkowski metric to be used to calculate distance between points. If `None`, then
            `mink_power=2` (equivalent to the Euclidean distance).
        :type mink_power: `float`, `optional`, defaults to `2`

        :param leaf_size:
            Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well
            as the memory required to store the tree. The optimal value depends on the nature of the problem.
        :type leaf_size: `int`, `optional`, defaults to `30`

        :param metric:
            The metric to use when calculating distance between instances in a feature array. If metric is a string or
            callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric
            parameter. If metric is `'precomputed'`, X is assumed to be a distance matrix and must be square. X may be
            a sparse graph, in which case only `'nonzero'` elements may be considered neighbors for DBSCAN.
        :type metric: `str`, or `callable`, default to `’euclidean’`


        :param scaling:
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a ClusteredDataset that knows the data and their labels
        :rtype: ClusteredDataset
        """
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size, metric=metric)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_pareto_sheets(cls, h_list: Dataset, costs_columns: List[str], nb_sheets: int = 1):
        """
        Get successive pareto sheets where each label is the index of a pareto sheet put them in a `ClusteredDataset`.

        A pareto sheet is defined as the optimal points in a DOE for a pre-computed costs.

        :param h_list:
            The Dataset in which to pick optimal points.
        :type h_list: Dataset

         :param costs_columns:
             List of columns' indexes or attributes on which costs are stored in current Dataset
         :type costs_columns: `List[str]`

        :param nb_sheets:
            Number of pareto sheets to pick
        :type nb_sheets: `int`, `optional`, default to `1`

        :return: a ClusteredDataset where each element is labelled with its pareto_sheet. Elements outside a
        pareto_sheet are labelled `n_sheets`
        :rtype: ClusteredDataset
        """
        labels = []
        dessia_objects = []
        pareto_sheets, non_optimal_points = h_list.pareto_sheets(costs_columns, nb_sheets)
        for label, pareto_sheet in enumerate(pareto_sheets):
            labels.extend([label] * len(pareto_sheet))
            dessia_objects.extend(pareto_sheet)
        dessia_objects.extend(non_optimal_points)
        labels.extend([len(pareto_sheets)] * len(non_optimal_points))
        return cls(dessia_objects, labels)

    @staticmethod
    def fit_cluster(skl_cluster: cluster, matrix: List[List[float]], scaling: bool):
        """
        Find clusters in data set for skl_cluster model.

        :param skl_cluster: sklearn.cluster object to compute clusters.
        :type data: cluster

        :param matrix:
            List of data
        :type matrix: `float`, `n_samples x n_features`

        :param scaling:
            Whether to scale the data or not before clustering.
        :type scaling: `bool`, `optional`, defaults to `False`

        :return: a fit sklearn.cluster object
        :rtype: cluster
        """
        if scaling:
            scaled_matrix = Dataset._scale_data(matrix)
        else:
            scaled_matrix = matrix
        skl_cluster.fit(scaled_matrix)
        return skl_cluster

    @classmethod
    def list_agglomerative_clustering(cls, data: List[DessiaObject], n_clusters: int = 2,
                                      metric: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None, scaling: bool = False, name: str = ""):
        """ Does the same as `from_agglomerative_clustering` method but data is a `List[DessiaObject]`. """
        return cls.from_agglomerative_clustering(Dataset(data), n_clusters=n_clusters, metric=metric,
                                                 linkage=linkage, distance_threshold=distance_threshold,
                                                 scaling=scaling, name=name)

    @classmethod
    def list_kmeans(cls, data: List[DessiaObject], n_clusters: int = 2, n_init: int = 10, tol: float = 1e-4,
                    scaling: bool = False, name: str = ""):
        """ Does the same as `from_kmeans` method but data is a `List[DessiaObject]`. """
        return cls.from_kmeans(Dataset(data), n_clusters=n_clusters, n_init=n_init, tol=tol, scaling=scaling,
                               name=name)

    @classmethod
    def list_dbscan(cls, data: List[DessiaObject], eps: float = 0.5, min_samples: int = 5, mink_power: float = 2,
                    leaf_size: int = 30, metric: str = "euclidean", scaling: bool = False, name: str = ""):
        """ Does the same as `from_dbscan` method but data is a `List[DessiaObject]`. """
        return cls.from_dbscan(Dataset(data), eps=eps, min_samples=min_samples, mink_power=mink_power,
                               leaf_size=leaf_size, metric=metric, scaling=scaling, name=name)

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
