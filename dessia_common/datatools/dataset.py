""" Library for building Dataset. """
from typing import List, Dict, Any
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
from sklearn import preprocessing

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle, Sample
    from plot_data.colors import BLUE, GREY
except ImportError:
    pass
from dessia_common.core import DessiaObject, DessiaFilter, FiltersList
from dessia_common.exports import MarkdownWriter
from dessia_common import templates
from dessia_common.datatools.metrics import mean, std, variance, covariance_matrix


class Dataset(DessiaObject):
    """
    Base object for handling a list of DessiaObjects.

    :param dessia_objects:
        --------
        List of DessiaObjects to store in Dataset
    :type dessia_objects: List[DessiaObject], `optional`, defaults to `None`

    :param name:
        --------
        Name of Dataset
    :type name: str, `optional`, defaults to `''`

    :Properties:
        * **common_attributes:** (`List[str]`)
            --------
            Common attributes of DessiaObjects contained in the current `Dataset`

        * **matrix:** (`List[List[float]]`, `n_samples x n_features`)
            --------
            Matrix of data computed by calling the `to_vector` method of all `dessia_objects`

    **Built-in methods**:
        * __init__
            >>> from dessia_common.datatools.dataset import Dataset
            >>> from dessia_common.models import all_cars_wi_feat
            >>> hlist = Dataset(all_cars_wi_feat, name="init")

        * __str__
            >>> print(Dataset(all_cars_wi_feat[:3], name='printed'))
            Dataset printed: 3 samples, 5 features
            |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
            -----------------------------------------------------------------------------------------------------------
            |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
            |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
            |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |

        * __len__
            >>> len(Dataset(all_cars_wi_feat))
            returns len(all_cars_wi_feat)

        * __get_item__
            >>> Dataset(all_cars_wi_feat)[0]
            returns <dessia_common.tests.CarWithFeatures object at 'memory_address'>
            >>> Dataset(all_cars_wi_feat)[0:2]
            returns Dataset(all_cars_wi_feat[0:2])
            >>> Dataset(all_cars_wi_feat)[[0,5,6]]
            returns Dataset([all_cars_wi_feat[idx] for idx in [0,5,6]])
            >>> booleans_list = [True, False,..., True] of length len(all_cars_wi_feat)
            >>> Dataset(all_cars_wi_feat)[booleans_list]
            returns Dataset([car for car, boolean in zip(all_cars_wi_feat, booleans_list) if boolean])

        * __add__
            >>> Dataset(all_cars_wi_feat) + Dataset(all_cars_wi_feat)
            Dataset(all_cars_wi_feat + all_cars_wi_feat)
            >>> Dataset(all_cars_wi_feat) + Dataset()
            Dataset(all_cars_wi_feat)
            >>> Dataset(all_cars_wi_feat).extend(Dataset(all_cars_wi_feat))
            Dataset(all_cars_wi_feat + all_cars_wi_feat)
    """

    _standalone_in_db = True
    _vector_features = ["name", "common_attributes"]
    _non_data_eq_attributes = ["name", "_common_attributes", "_matrix"]

    def __init__(self, dessia_objects: List[DessiaObject] = None, name: str = ''):
        """ See class docstring. """
        if dessia_objects is None:
            dessia_objects = []
        self.dessia_objects = dessia_objects
        self._common_attributes = None
        self._matrix = None
        DessiaObject.__init__(self, name=name)

    def __getitem__(self, key: Any):
        """
        Custom getitem for Dataset.

        In addition to work as numpy.arrays of dimension `(n,)`, allows to pick a sub-Dataset from a list of indexes.
        """
        if len(self) == 0:
            return []
        if isinstance(key, int):
            return self._pick_from_int(key)
        if isinstance(key, slice):
            return self._pick_from_slice(key)
        if isinstance(key, list):
            if len(key) == 0:
                return self.__class__()
            if isinstance(key[0], bool):
                if len(key) == self.__len__():
                    return self._pick_from_boolist(key)
                raise ValueError(f"Cannot index {self.__class__.__name__} object of len {self.__len__()} with a "
                                 f"list of boolean of len {len(key)}")
            if isinstance(key[0], int):
                return self._pick_from_boolist(self._indexlist_to_booleanlist(key))

            raise NotImplementedError(f"key of type {type(key)} with {type(key[0])} elements not implemented for "
                                      f"indexing Datasets")

        raise NotImplementedError(f"key of type {type(key)} not implemented for indexing Datasets")

    def __add__(self, other: 'Dataset'):
        """ Allows to merge two Dataset into one by merging their dessia_object into one list. """
        if self.__class__ != Dataset or other.__class__ != Dataset:
            raise TypeError("Addition only defined for Dataset. A specific __add__ method is required for "
                            f"{self.__class__}")

        sum_hlist = self.__class__(dessia_objects=self.dessia_objects + other.dessia_objects,
                                   name=self.name[:5] + '_+_' + other.name[:5])

        if all(item in self.common_attributes for item in other.common_attributes):
            sum_hlist._common_attributes = self.common_attributes
            if self._matrix is not None and other._matrix is not None:
                sum_hlist._matrix = self._matrix + other._matrix
        return sum_hlist

    def extend(self, other: 'Dataset'):
        """
        Update a Dataset by adding b values to it.

        :param b: Dataset to add to the current Dataset
        :type b: Dataset

        :return: None

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> Dataset(all_cars_wi_feat).extend(Dataset(all_cars_wi_feat))
        Dataset(all_cars_wi_feat + all_cars_wi_feat)
        """
        # Not "self.dessia_objects += other.dessia_objects" to take advantage of __add__ algorithm
        self.__dict__.update((self + other).__dict__)

    def _pick_from_int(self, idx: int):
        return self.dessia_objects[idx]

    def _pick_from_slice(self, key: slice):
        new_hlist = self.__class__(dessia_objects=self.dessia_objects[key], name=self.name)
        new_hlist._common_attributes = copy(self._common_attributes)
        new_hlist.dessia_objects = self.dessia_objects[key]
        if self._matrix is not None:
            new_hlist._matrix = self._matrix[key]
        # new_hlist.name += f"_{key.start if key.start is not None else 0}_{key.stop}")
        return new_hlist

    def _indexlist_to_booleanlist(self, index_list: List[int]):
        boolean_list = [False] * len(self)
        for idx in index_list:
            boolean_list[idx] = True
        return boolean_list

    def _pick_from_boolist(self, key: List[bool]):
        new_hlist = self.__class__(dessia_objects=DessiaFilter.apply(self.dessia_objects, key), name=self.name)
        new_hlist._common_attributes = copy(self._common_attributes)
        if self._matrix is not None:
            new_hlist._matrix = DessiaFilter.apply(self._matrix, key)
        # new_hlist.name += "_list")
        return new_hlist

    def __str__(self):
        """ Print Dataset as a table. """
        attr_space = []

        prefix = self._write_str_prefix()

        if self.__len__() == 0:
            return prefix

        string = ""
        string += self._print_titles(attr_space)
        string += "\n" + "-" * len(string)
        string += self._print_objects_slice(slice(0, 5), attr_space)

        if len(self) > 10:
            undispl_len = len(self) - 10
            string += (f"\n+ {undispl_len} undisplayed object" + "s" * (min([undispl_len, 2]) - 1) + "...")

        if len(self) > 5:
            string += self._print_objects_slice(slice(-5, len(self)), attr_space)
        return prefix + "\n" + string + "\n"

    def _printed_attributes(self):
        if 'name' in self.common_attributes:
            return self.common_attributes
        return ['name'] + self.common_attributes

    def _print_objects_slice(self, key: slice, attr_space: List[int]):
        string = ""
        for index in range(len(self[key])):
            string += "\n"
            string += self._print_object(index, attr_space)
        return string

    def _write_str_prefix(self):
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += f"{len(self)} samples, {len(self.common_attributes)} features"
        return prefix

    def _print_titles(self, attr_space: List[int]):
        min_col_length = 16
        printed_attributes = self._printed_attributes()
        string = ""
        for idx, attr in enumerate(printed_attributes):
            end_bar = ""
            if idx == len(printed_attributes) - 1:
                end_bar = "|"
            # attribute
            if len(attr) + 6 > min_col_length:
                indentation = 3
            else:
                indentation = min_col_length - len(attr)
                odd_incr = int(indentation % 2)
                indentation = int(indentation / 2)

            name_attr = " " * indentation + " " * odd_incr + f"{attr.capitalize()}" + " " * indentation
            attr_space.append(len(name_attr))
            string += "|" + name_attr + end_bar
        return string

    def _print_object(self, index: int, attr_space: List[int]):
        printed_attributes = self._printed_attributes()
        string = ""
        for idx, attr in enumerate(printed_attributes):
            end_bar = ""
            if idx == len(printed_attributes) - 1:
                end_bar = "|"

            attr_value = self._get_printed_value(index, attr)

            string += "|" + " " * max((attr_space[idx] - len(str(attr_value)) - 1), 1)
            string += f"{attr_value}"[:attr_space[idx] - 4]
            if len(str(attr_value)) > attr_space[idx] - 3:
                string = string[:-1] + "... "
            elif len(str(attr_value)) == attr_space[idx] - 3:
                string += f"{attr_value}"[-1] + " "
            else:
                string += " "
            string += end_bar
        return string

    def to_markdown(self) -> str:
        """Render a markdown of the object output type: string."""
        md_writer = MarkdownWriter(print_limit=25, table_limit=12)
        name = md_writer.print_name(self)
        class_ = md_writer.print_class(self)
        element_details = md_writer.element_details(self)
        table = md_writer.matrix_table(self.matrix, self.common_attributes)

        return templates.dataset_markdown_template.substitute(name=name, class_=class_, element_details=element_details,
                                                              table=table)

    def _get_printed_value(self, index: int, attr: str):
        try:
            return getattr(self[index], attr)
        except AttributeError:
            return self.matrix[index][self.common_attributes.index(attr)]

    def __len__(self):
        """Length of Dataset is len(Dataset.dessia_objects)."""
        return len(self.dessia_objects)

    @property
    def common_attributes(self):
        """List of common attributes of stored dessia_objects."""
        if self._common_attributes is None:
            if len(self) == 0:
                return []

            all_class = []
            one_instance = []
            for dessia_object in self.dessia_objects:
                if dessia_object.__class__ not in all_class:
                    all_class.append(dessia_object.__class__)
                    one_instance.append(dessia_object)

            all_attributes = sum((instance.vector_features() for instance in one_instance), [])
            set_attributes = set.intersection(*(set(instance.vector_features()) for instance in one_instance))

            # Keep order
            self._common_attributes = []
            for attr in all_attributes:
                if attr in set_attributes:
                    self._common_attributes.append(attr)
                    set_attributes.remove(attr)

        return self._common_attributes

    @property
    def matrix(self):
        """
        Get equivalent matrix of dessia_objects.

        Dimensions: `len(dessia_objects) x len(common_attributes)`.
        """
        if self._matrix is None:
            matrix = []
            for dessia_object in self.dessia_objects:
                temp_row = dessia_object.to_vector()
                vector_features = dessia_object.vector_features()
                matrix.append(list(temp_row[vector_features.index(attr)] for attr in self.common_attributes))
            self._matrix = matrix
        return self._matrix

    def attribute_values(self, attribute: str):
        """
        Get a list of all values of dessia_objects of an attribute given by name.

        :param attribute: Attribute to get all values
        :type attribute: str

        :return: A list of all values of the specified attribute of dessia_objects
        :rtype: List[Any]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> Dataset(all_cars_wi_feat[:10]).attribute_values("weight")
        [3504.0, 3693.0, 3436.0, 3433.0, 3449.0, 4341.0, 4354.0, 4312.0, 4425.0, 3850.0]
        """
        if not hasattr(self.dessia_objects[0], attribute):
            if attribute not in self.common_attributes:
                raise ValueError(f"{attribute} not in common_attributes = {self.common_attributes}")
            return self.column_values(self.common_attributes.index(attribute))
        return [getattr(dessia_object, attribute) for dessia_object in self.dessia_objects]

    def column_values(self, index: int):
        """
        Get a list of all values of dessia_objects for an attribute given by its index common_attributes.

        :param index: Index in common_attributes to get all values of dessia_objects
        :type index: int

        :return: A list of all values of the specified attribute of dessia_objects
        :rtype: List[float]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> Dataset(all_cars_wi_feat[:10]).column_values(2)
        [130.0, 165.0, 150.0, 150.0, 140.0, 198.0, 220.0, 215.0, 225.0, 190.0]
        """
        return [row[index] for row in self.matrix]

    def sub_matrix(self, columns_names: List[str]):
        """
        Build a sub matrix of the current Dataset taking column numbers in indexes or attribute values in attributes.

        Warning: Only one of `indexes` or `attributes` has to be specified.

        :param columns_names: List of columns' names to create a sub matrix
        :type columns_names: List[str]

        :return: Data stored in matrix reduced to the specified `indexes` or `attributes`
        :rtype: List[List[float]]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> print(Dataset(all_cars_wi_feat[:10]).sub_matrix(['displacement', 'horsepower']))
        [[0.307, 130.0], [0.35, 165.0], [0.318, 150.0], [0.304, 150.0], [0.302, 140.0], [0.429, 198.0],
         [0.454, 220.0], [0.44, 215.0], [0.455, 225.0], [0.39, 190.0]]
        """
        transposed_submatrix = [self.attribute_values(column_name) for column_name in columns_names]
        return list(map(list, zip(*transposed_submatrix)))

    def sort(self, key: Any, ascend: bool = True):  # TODO : Replace numpy with faster algorithms
        """
        Sort the current Dataset along the given key.

        :param key:
            --------
            The parameter on which to sort the Dataset. Can be an attribute or its index in \
                `common_attributes`
        :type key: int or str

        :param ascend:
            --------
            Whether to sort the Dataset in ascending (`True`) or descending (`False`) order
        :type key: bool, defaults to `True`

        :return: None

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat[:3], "sort_example")
        >>> example_list.sort("mpg", False)
        >>> print(example_list)
        Dataset sort_example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |
        |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
        |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
        >>> example_list.sort(2, True)
        >>> print(example_list)
        Dataset sort_example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
        |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |
        |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
        """
        if len(self) != 0:
            if isinstance(key, int):
                sort_indexes = npy.argsort(self.column_values(key))
            elif isinstance(key, str):
                sort_indexes = npy.argsort(self.attribute_values(key))
            self.dessia_objects = [self.dessia_objects[idx] for idx in (sort_indexes if ascend else sort_indexes[::-1])]
            if self._matrix is not None:
                self._matrix = [self._matrix[idx] for idx in
                                (sort_indexes if ascend else sort_indexes[::-1])]

    def mean(self):
        """
        Compute means along each `common_attribute`.

        :return: A list of means along each dimension
        :rtype: List[float]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat, "mean_example")
        >>> print(example_list.mean())
        [23.051231527093602, 0.1947795566502462, 103.5295566502463, 2979.4137931034484, 15.519704433497521]
        """
        return [mean(row) for row in zip(*self.matrix)]

    def standard_deviation(self):
        """
        Compute standard deviations along each `common_attribute`.

        :return: A list of standard deviations along each dimension
        :rtype: List[float]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat, "std_example")
        >>> print(example_list.standard_deviation())
        [8.391423956652817, 0.10479316386533469, 40.47072606559397, 845.9605763601298, 2.799904275515381]
        """
        return [std(row) for row in zip(*self.matrix)]

    def variances(self):
        """
        Compute variances along each `common_attribute`.

        :return: A list of variances along each dimension
        :rtype: List[float]

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat, "var_example")
        >>> print(example_list.variances())
        [70.41599602028683, 0.010981607192906888, 1637.8796682763475, 715649.2967555631, 7.839463952049309]
        """
        return [variance(row) for row in zip(*self.matrix)]

    def covariance_matrix(self):
        """
        Compute the covariance matrix of `self.matrix`.

        :return: the covariance matrix of all stored data in self
        :rtype: List[List[float]], `n_features x n_features`

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat, "covar_example")
        >>> cov_matrix = example_list.covariance_matrix()
        >>> for row in cov_matrix: print(row)
        [70.58986267712706, -0.6737370735267286, -247.39164142796338, -5604.189893571734, 9.998099130329008]
        [-0.6737370735267286, 0.011008722272395539, 3.714807148938756, 82.86881366538952, -0.16412268260049875]
        [-247.39164142796338, 3.714807148938756, 1641.9238156054248, 28857.60749255002, -77.47638630420236]
        [-5604.189893571734, 82.86881366538952, 28857.60749255002, 717416.332056194, -1021.2202724563649]
        [9.998099130329008, -0.16412268260049875, -77.47638630420236, -1021.2202724563649, 7.8588206531654805]
        """
        return covariance_matrix(list(zip(*self.matrix)))

    def distance_matrix(self, method: str = 'minkowski', **kwargs):
        """
        Compute the distance matrix of `self.matrix`, i.e. the pairwise distances between all dessia_objects.

        Distances are computed with numerical values of `self.matrix`.

        :param method:
            Method to compute distances.
            Can be one of `[‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, \
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, \
            ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]`.
        :type method: str, `optional`, defaults to `'minkowski'`

        :param **kwargs:
            |  Extra arguments to metric: refer to each metric documentation for a list of all possible arguments.
            |  Some possible arguments:
            |     - p : scalar The p-norm to apply for Minkowski, weighted and unweighted. Default: `2`.
            |     - w : array_like The weight vector for metrics that support weights (e.g., Minkowski).
            |     - V : array_like The variance vector for standardized Euclidean. Default: \
                `var(vstack([XA, XB]), axis=0, ddof=1)`
            |     - VI : array_like The inverse of the covariance matrix for Mahalanobis. Default: \
                `inv(cov(vstack([XA, XB].T))).T`
            |     - out : ndarray The output array If not None, the distance matrix Y is stored in this array.
        :type **kwargs: dict, `optional`

        :return: the distance matrix of all stored data in self
        :rtype: List[List[float]], `n_samples x n_samples`

        :Examples:
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = Dataset(all_cars_wi_feat, "distance_example")
        >>> distance_matrix = example_list.distance_matrix('mahalanobis')
        >>> for row in distance_matrix[:4]: print(row[:4])
        [0.0, 1.6150355142162274, 1.0996902429379676, 1.3991408510938068]
        [1.6150355142162274, 0.0, 0.7691239946132247, 0.6216479207905371]
        [1.0996902429379676, 0.7691239946132247, 0.0, 0.7334135920381655]
        [1.3991408510938068, 0.6216479207905371, 0.7334135920381655, 0.0]
        """
        kwargs = self._set_distance_kwargs(method, kwargs)
        distances = squareform(pdist(self.matrix, method, **kwargs)).astype(float)
        return distances.tolist()

    @staticmethod
    def _set_distance_kwargs(method: str, kwargs: Dict[str, Any]):
        if 'p' not in kwargs and method == 'minkowski':
            kwargs['p'] = 2
        return kwargs

    def filtering(self, filters_list: FiltersList):
        """
        Filter a Dataset given a FiltersList. Method filtering apply a FiltersList to the current Dataset.

        :param filters_list:
            FiltersList to apply on current Dataset
        :type filters_list: FiltersList

        :return: The filtered Dataset
        :rtype: Dataset

        :Examples:
        >>> from dessia_common.core import DessiaFilter
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> filters = [DessiaFilter('weight', '<=', 1650.), DessiaFilter('mpg', '>=', 45.)]
        >>> filters_list = FiltersList(filters, "xor")
        >>> example_list = Dataset(all_cars_wi_feat, name="example")
        >>> filtered_list = example_list.filtering(filters_list)
        >>> print(filtered_list)
        Dataset example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               35.0  |             0.072  |              69.0  |            1613.0  |              18.0  |
        |               31.0  |             0.076  |              52.0  |            1649.0  |              16.5  |
        |               46.6  |             0.086  |              65.0  |            2110.0  |              17.9  |
        """
        booleans_index = filters_list.get_booleans_index(self.dessia_objects)
        return self._pick_from_boolist(booleans_index)

    def singular_values(self):
        """
        Computes the Singular Values Decomposition (SVD) of self.matrix.

        SVD factorizes self.matrix into two unitary matrices `U` and `Vh`, and a 1-D array `s` of singular values \
            (real, non-negative) such that ``a = U @ S @ Vh``, where S is diagonal such as `s1 > s2 >...> sn`.

        SVD gives indications on the dimensionality of a given matrix thanks to the normalized singular values: they \
            are stored in descending order and their sum is equal to 1. Thus, one can set a threshold value, e.g. \
                `0.95`, and keep only the `r` first normalized singular values which sum is greater than the threshold.

        `r` is the rank of the matrix and gives a good indication on the real dimensionality of the data contained in \
            the current Dataset. `r` is often much smaller than the current dimension of the studied data.
        This indicates that the used features can be combined into less new features, which do not necessarily \
            make sense for engineers.

        More informations: https://en.wikipedia.org/wiki/Singular_value_decomposition

        :return:
            **normalized_singular_values**: list of normalized singular values
            **singular_points**: list of points to plot in dimensionality plot. Does not add any information.
        :rtype: Tuple[List[float], List[Dict[str, float]]]
        """
        scaled_data = Dataset._scale_data(npy.array(self.matrix) - npy.mean(self.matrix, axis=0))
        _, singular_values, _ = npy.linalg.svd(npy.array(scaled_data).T, full_matrices=False)
        normalized_singular_values = singular_values / npy.sum(singular_values)

        singular_points = []
        for idx, value in enumerate(normalized_singular_values):
            # TODO (plot_data log_scale 0)
            singular_points.append({'Index of reduced basis vector': idx + 1,
                                    'Singular value': (value if value != 0. else 1e-16)})
        return normalized_singular_values, singular_points

    @staticmethod
    def _scale_data(data_matrix: List[List[float]]):
        scaled_matrix = preprocessing.StandardScaler().fit_transform(data_matrix)
        return [list(map(float, row.tolist())) for row in scaled_matrix]

    def plot_data(self, reference_path: str = "#", **kwargs):
        """ Plot a standard scatter matrix of all attributes in common_attributes and a dimensionality plot. """
        data_list = self._to_samples(reference_path=reference_path)
        if len(self.common_attributes) > 1:
            # Plot a correlation matrix : To develop
            # correlation_matrix = []
            # Dimensionality plot
            dimensionality_plot = self._plot_dimensionality()
            # Scattermatrix
            scatter_matrix = self._build_multiplot(data_list, self._tooltip_attributes(), axis=dimensionality_plot.axis,
                                                   point_style=dimensionality_plot.point_style)
            # Parallel plot
            parallel_plot = self._parallel_plot(data_list)
            return [parallel_plot, scatter_matrix]  # , dimensionality_plot]

        plot_mono_attr = self._histogram_unic_value(0, name_attr=self.common_attributes[0])
        plot_mono_attr.elements = data_list
        return [plot_mono_attr]

    def _build_multiplot(self, data_list: List[Dict[str, float]], tooltip: List[str], **kwargs: Dict[str, Any]):
        subplots = []
        for line in self.common_attributes:
            for idx_col, col in enumerate(self.common_attributes):
                if line == col:
                    subplots.append(self._histogram_unic_value(idx_col, col))
                else:
                    subplots.append(Scatter(x_variable=line, y_variable=col, tooltip=Tooltip(tooltip), **kwargs))

        scatter_matrix = MultiplePlots(plots=subplots, elements=data_list, point_families=self._point_families(),
                                       initial_view_on=True)
        return scatter_matrix

    def _histogram_unic_value(self, idx_col: int, name_attr: str):
        # unic_values = set((getattr(dobject, line) for dobject in self.dessia_objects))
        unic_values = set((row_matrix[idx_col] for row_matrix in self.matrix))
        if len(unic_values) == 1:  # TODO (plot_data linspace axis between two same values)
            plot_obj = Scatter(x_variable=name_attr, y_variable=name_attr)
        else:
            plot_obj = Histogram(x_variable=name_attr)
        return plot_obj

    def _tooltip_attributes(self):
        return self.common_attributes

    def _object_to_sample(self, dessia_object: DessiaObject, row: int, reference_path: str = '#'):
        sample_values = {attr: self.matrix[row][col] for col, attr in enumerate(self.common_attributes)}
        reference_path = f"{reference_path}/dessia_objects/{row}"
        name = dessia_object.name if dessia_object.name else f"Sample {row}"
        return Sample(values=sample_values, reference_path=reference_path, name=name)

    def _to_samples(self, reference_path: str = '#'):
        return [self._object_to_sample(dessia_object=dessia_object, row=row, reference_path=reference_path)
                for row, dessia_object in enumerate(self.dessia_objects)]

    def _point_families(self):
        return [PointFamily(BLUE, list(range(len(self))))]

    def _parallel_plot(self, data_list: List[Dict[str, float]]):
        return ParallelPlot(elements=data_list, axes=self._parallel_plot_attr(), disposition='vertical')

    def _parallel_plot_attr(self):
        # TODO: Put it in plot_data
        (sorted_r2, sorted_association), constant_attributes = self._get_correlations()
        attribute_series = self._get_attribute_trios(sorted_r2, sorted_association)
        return constant_attributes + self._trios_list_to_parallel_axes(attribute_series)

    def _get_correlations(self):
        r2_scores = []
        association_list = []
        constant_attributes = []
        for idx, attr1 in enumerate(self.common_attributes):
            if len(set(self.attribute_values(attr1))) == 1:
                constant_attributes.append(attr1)
                continue
            for _, attr2 in enumerate(self.common_attributes[idx:]):
                if len(set(self.attribute_values(attr2))) == 1:
                    constant_attributes.append(attr2)
                    continue
                if attr1 != attr2:
                    correlation_matrix = npy.corrcoef(self.attribute_values(attr1),
                                                      self.attribute_values(attr2))
                    correlation_xy = correlation_matrix[0, 1]
                    r2_scores.append(correlation_xy**2)
                    association_list.append([attr1, attr2])

        if len(association_list) == 0:
            association_list = []
            r2_scores = []
            unreal_score = 1.
            for idx, attr1 in enumerate(self.common_attributes):
                for attr2 in self.common_attributes[idx + 1:]:
                    association_list.append([attr1, attr2])
                    r2_scores.append(unreal_score)
                    unreal_score += -1 / 10.
            return map(list, zip(*sorted(zip(r2_scores, association_list))[::-1])), []
        # Returns list of list of associated attributes sorted along their R2 score and constant attributes
        return map(list, zip(*sorted(zip(r2_scores, association_list))[::-1])), list(set(constant_attributes))

    @staticmethod
    def _get_attribute_trios(sorted_r2, sorted_association):
        attribute_series = []
        picked_attr = set()
        set_association = set(sum(sorted_association, []))
        while len(picked_attr) != len(set_association):
            first_association = sorted_association[0]
            attribute_series.append(first_association)
            picked_attr.update(set(first_association))
            del sorted_r2[0], sorted_association[0]

            for idx, _ in enumerate(sorted_r2):
                if any(item in first_association for item in sorted_association[idx]):
                    attribute_series[-1] += sorted_association[idx]
                    picked_attr.update(set(sorted_association[idx]))
                    del sorted_r2[idx], sorted_association[idx]
                    break
        return attribute_series

    def _trios_list_to_parallel_axes(self, attribute_series):
        ordered_attr = []
        for attribute_serie in attribute_series:
            if not any(item in attribute_serie for item in ordered_attr):
                ordered_attr += self._new_attributes_trio(attribute_serie)
            else:
                ordered_attr = self._new_sided_attribute(ordered_attr, attribute_serie)
        return ordered_attr

    @staticmethod
    def _new_attributes_trio(attribute_serie):
        if len(attribute_serie) < 3:
            return attribute_serie
        mid_index = [attribute_serie.count(attr) for attr in attribute_serie].index(2)
        mid_attr = attribute_serie[mid_index]
        remaining_attr = iter(set(attribute_serie).difference({mid_attr}))
        return [next(remaining_attr), mid_attr, next(remaining_attr)]

    @staticmethod
    def _new_sided_attribute(ordered_attr, attribute_serie):
        for side in [0, -1]:
            if ordered_attr[side] in attribute_serie:
                nb_instances = attribute_serie.count(ordered_attr[side])
                for ieme_instance in range(nb_instances):
                    idx_in_serie = (ieme_instance) * 2 + attribute_serie[ieme_instance * 2:].index(ordered_attr[side])
                    # 1 if idx_in_serie = 0, 0 if idx_in_serie = 1, 3 if idx_in_serie = 2, 2 if idx_in_serie = 3
                    idx_attr_to_add = idx_in_serie + 1 - 2 * (idx_in_serie % 2)
                    added_attr = []
                    if attribute_serie[idx_attr_to_add] not in ordered_attr:
                        added_attr = [attribute_serie[idx_attr_to_add]]
                        ordered_attr = (side + 1) * added_attr + ordered_attr + (-1 * side) * added_attr
                        break
        return ordered_attr

    def _plot_dimensionality(self):
        _, singular_points = self.singular_values()

        axis_style = EdgeStyle(line_width=0.5, color_stroke=GREY)
        axis = Axis(nb_points_x=len(singular_points), nb_points_y=len(singular_points), axis_style=axis_style)
        point_style = PointStyle(color_fill=BLUE, color_stroke=BLUE, stroke_width=0.1, size=2, shape='circle')

        dimensionality_plot = Scatter(elements=singular_points, x_variable='Index of reduced basis vector',
                                      y_variable='Singular value', log_scale_y=True, axis=axis, point_style=point_style)
        return dimensionality_plot

    @staticmethod
    def _check_costs(len_data: int, costs: List[List[float]]):
        if len(costs) != len_data:
            if len(costs[0]) == len_data:
                return list(map(list, zip(*costs)))
            raise ValueError(f"costs is length {len(costs)} and the matching Dataset is length {len_data}. " +
                             "They should be the same length.")
        return costs

    @staticmethod
    def pareto_indexes(costs: List[List[float]]):
        """
        Find the Pareto-efficient points.

        :return: A (n_points, ) boolean list, indicating whether each point is Pareto efficient
        """
        is_efficient = npy.ones(len(costs), dtype=bool)
        costs_array = (costs - npy.mean(costs, axis=0)) / npy.std(costs, axis=0)
        for index, cost in enumerate(costs_array):
            if is_efficient[index]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = npy.any(costs_array[is_efficient] < cost, axis=1)
                # And keep self
                is_efficient[index] = True
        return is_efficient.tolist()

    @staticmethod
    def pareto_frontiers(len_data: int, costs: List[List[float]]):
        """ Experimental method to draw the borders of Pareto domain. """
        checked_costs = Dataset._check_costs(len_data, costs)
        pareto_indexes = Dataset.pareto_indexes(checked_costs)
        pareto_costs = npy.array(list(itertools.compress(checked_costs, pareto_indexes)))

        array_costs = npy.array(checked_costs)
        super_mini = npy.min(array_costs, axis=0)
        pareto_frontiers = []
        for x_dim in range(pareto_costs.shape[1]):
            for y_dim in range(pareto_costs.shape[1]):
                if x_dim != y_dim:
                    frontier_2d = Dataset._pareto_frontier_2d(x_dim, y_dim, pareto_costs,
                                                              npy.max(array_costs[:, x_dim]), super_mini)
                    pareto_frontiers.append(frontier_2d)
        return pareto_frontiers

    @staticmethod
    def _pareto_frontier_2d(x_dim: int, y_dim: int, pareto_costs: List[List[float]], max_x_dim: float,
                            super_mini: List[float]):
        # Experimental
        minidx = npy.argmin(pareto_costs[:, y_dim])
        x_coord = pareto_costs[minidx, x_dim]
        y_coord = pareto_costs[minidx, y_dim]

        new_pareto = pareto_costs[x_coord - pareto_costs[:, x_dim] != 0., :]

        dir_coeffs = (y_coord - new_pareto[:, y_dim]) / (x_coord - new_pareto[:, x_dim])
        dir_coeffs[x_coord == new_pareto[:, x_dim]] = npy.max(dir_coeffs[x_coord != new_pareto[:, x_dim]])

        offsets = y_coord - dir_coeffs * x_coord
        approx_super_mini = dir_coeffs * super_mini[x_dim] + offsets
        chosen_line = npy.argmin(npy.absolute(approx_super_mini - super_mini[y_dim]))

        frontier_2d = npy.array([[super_mini[x_dim], max_x_dim], [approx_super_mini[chosen_line], max_x_dim *
                                                                  dir_coeffs[chosen_line] + offsets[chosen_line]]]).T
        return frontier_2d

    def _compute_costs(self, costs_attributes: List[str]):
        costs = self.sub_matrix(costs_attributes)
        return Dataset._check_costs(len(self), costs)

    def pareto_points(self, costs_attributes: List[str]):
        """
        Find the Pareto-efficient points.

        :param costs_attributes:
            List of columns' attributes on which costs are stored in current Dataset
        :type costs_attributes: List[str]

        :return: a Dataset containing the selected points
        :rtype: Dataset
        """
        checked_costs = self._compute_costs(costs_attributes)
        return self[self.__class__.pareto_indexes(checked_costs)]

    def pareto_sheets(self, costs_attributes: List[str], nb_sheets: int = 1):
        """
        Get successive Pareto sheets (i.e. optimal points in a DOE for pre-computed costs).

        :param costs_attributes: List of columns' attributes on which costs are stored in current Dataset
        :type costs_attributes: List[str]

        :param nb_sheets: Number of Pareto sheets to pick
        :type nb_sheets: int, `optional`, default to `1`

        :return: The successive Pareto sheets and not selected elements
        :rtype: List[Dataset], Dataset
        """
        checked_costs = self._compute_costs(costs_attributes)
        non_optimal_costs = checked_costs[:]
        non_optimal_points = self.dessia_objects[:]
        pareto_sheets = []
        for idx in range(nb_sheets):
            pareto_sheet = Dataset.pareto_indexes(non_optimal_costs)
            pareto_sheets.append(Dataset(list(itertools.compress(non_optimal_points, pareto_sheet)),
                                         self.name + f'_pareto_{idx}'))
            non_optimal_points = list(itertools.compress(non_optimal_points, map(lambda x: not x, pareto_sheet)))
            non_optimal_costs = list(itertools.compress(non_optimal_costs, map(lambda x: not x, pareto_sheet)))
        return pareto_sheets, Dataset(non_optimal_points, self.name)
