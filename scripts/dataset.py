"""
Tests for dessia_common.Dataset class (loadings, check_platform and plots).
"""
import random
from dessia_common.core import DessiaObject
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_middl
from dessia_common.datatools.math import covariance, manhattan_distance, euclidean_distance, minkowski_distance,\
    inf_norm, mahalanobis_distance
from dessia_common.datatools.dataset import Dataset

# Tests on common_attributes
class SubObject(DessiaObject):
    def __init__(self, sub_attr: float = 1.5, name: str = ''):
        self.sub_attr = sub_attr
        DessiaObject.__init__(self, name=name)


class TestObject(DessiaObject):
    _vector_features = ['attr1', "sub_object/sub_attr"]

    def __init__(self, attr1: float = 1.2, sub_object: SubObject = SubObject(), name: str = ''):
        self.attr1 = attr1
        self.attr2 = attr1 * 2
        self.sub_object = sub_object
        DessiaObject.__init__(self, name=name)

    @property
    def prop1(self):
        return self.attr1 + self.attr2


test_object = TestObject()
sub_object_dataset = Dataset([SubObject()] * 10)
test_dataset = Dataset([test_object] * 10)
test_dataset.plot_data()
assert(test_dataset.common_attributes == ['attr1', 'sub_object/sub_attr'])
assert("Sub_object/sub_attr" in test_dataset.__str__())
assert("Sub_attr" in sub_object_dataset.__str__())

# Tests on common_attributes
class TestObject(DessiaObject):
    _vector_features = ['attr1', 'attr2', 'prop1', 'in_to_vector']

    def __init__(self, attr1: float = 1.2, name: str = ''):
        self.attr1 = attr1
        self.attr2 = attr1 * 2
        DessiaObject.__init__(self, name=name)

    @property
    def prop1(self):
        return self.attr1 + self.attr2

    def to_vector(self):
        return [self.attr1, self.attr2, self.prop1, random.randint(0, 32)]


test_object = TestObject()
test_dataset = Dataset([test_object] * 10)
test_dataset.plot_data()
assert(all(value in test_dataset._print_object(6, [12, 12, 12, 12, 12]) for value in ["1.2", "2.4", "3.59999..."]))
assert(test_dataset.common_attributes == ['attr1', 'attr2', 'prop1', 'in_to_vector'])

# When attribute _features is not specified in class Car
all_cars_without_features = Dataset(all_cars_no_feat)

# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = Dataset(all_cars_wi_feat)
# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = Dataset(rand_data_middl)

# Compute one common_attributes
all_cars_without_features.common_attributes

# Compute features importances from RandomForest algorithm
input_attributes = ['displacement', 'horsepower', 'model', 'acceleration', 'cylinders']
output_attributes = ['weight']

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
RandData_heterogeneous._check_platform()

# Test __getitem__
picked_list = (all_cars_with_features[250:] +
               RandData_heterogeneous[:50][[1, 4, 6, 10, 25]][[True, False, True, True, False]])
assert(picked_list._common_attributes is None)
assert(picked_list._matrix is None)
assert(picked_list[-1] == rand_data_middl[10])
try:
    all_cars_without_features[[True, False, True]]
    raise ValueError("boolean indexes of len 3 should not be able to index Datasets of len 406")
except Exception as e:
    assert(e.args[0] == "Cannot index Dataset object of len 406 with a list of boolean of len 3")

# Test on matrice
idx = random.randint(0, len(all_cars_without_features) - 1)
assert(all(item in all_cars_without_features.matrix[idx]
           for item in [getattr(all_cars_without_features.dessia_objects[idx], attr)
                        for attr in all_cars_without_features.common_attributes]))

# Tests for displays
hlist_cars_plot_data = all_cars_without_features.plot_data()
assert(all(string in all_cars_with_features.__str__() for string in ["Chevrolet C...", "0.302 |", "+ 396 undisplayed"]))

# Tests for metrics
assert(int(all_cars_with_features.distance_matrix('minkowski')[25][151]) == 186)
assert(int(all_cars_with_features.mean()[3]) == 15)
assert(int(all_cars_with_features.standard_deviation()[4]) == 845)
assert(int(all_cars_with_features.variances()[2]) == 1637)
assert(int(manhattan_distance(all_cars_with_features.matrix[3], all_cars_with_features.matrix[125])) == 1361)
assert(int(minkowski_distance(all_cars_with_features.matrix[3],
       all_cars_with_features.matrix[125], mink_power=7.2)) == 1275)
assert(int(euclidean_distance(all_cars_with_features.matrix[3], all_cars_with_features.matrix[125])) == 1277)
assert(int(covariance(all_cars_with_features.matrix[3], all_cars_with_features.matrix[125])) == 1155762)
assert(int(inf_norm([1, 2, 3, 45, 4., 4.21515, -12, -0, 0, -25214.1511])) == 25214)
assert(int(mahalanobis_distance(all_cars_with_features.matrix[3],
                                all_cars_with_features.matrix[125],
                                all_cars_with_features.covariance_matrix())) == 2)
assert(all_cars_with_features.maximums == [46.6, 0.455, 230.0, 24.8, 5140.0])
assert(all_cars_with_features.minimums == [0.0, 0.068, 0.0, 8.0, 1613.0])

# Tests for empty Dataset
empty_list = Dataset()
print(empty_list)
assert(empty_list[0] == [])
assert(empty_list[:] == [])
assert(empty_list[[False, True]] == [])
assert(empty_list + empty_list == Dataset())
assert(empty_list + all_cars_without_features == all_cars_without_features)
assert(all_cars_without_features + empty_list == all_cars_without_features)
assert(len(empty_list) == 0)
assert(empty_list.matrix == [])
assert(empty_list.common_attributes == [])
empty_list.sort(0)
assert(empty_list == Dataset())
empty_list.sort("weight")
assert(empty_list == Dataset())

try:
    empty_list.plot_scatter_matrix()
    raise ValueError("plot_data should not work on empty Datasets")
except Exception as e:
    assert(e.__class__.__name__ == "ValueError")

try:
    empty_list.singular_values()
    raise ValueError("singular_values should not work on empty Datasets")
except Exception as e:
    assert(e.__class__.__name__ == "ValueError")

# Tests sort
all_cars_with_features.sort('weight', ascend=False)
assert(all_cars_with_features[0].weight == max(all_cars_with_features.attribute_values('weight')))

idx_dpl = all_cars_without_features.common_attributes.index('displacement')
all_cars_without_features.sort(idx_dpl)
assert(all(attr in ['displacement', 'cylinders', 'mpg', 'horsepower', 'weight', 'acceleration', 'model']
           for attr in all_cars_without_features.common_attributes))
assert(all_cars_without_features[0].displacement == min(all_cars_without_features.column_values(idx_dpl)))

# Missing tests after coverage report
assert(all_cars_without_features[[]] == empty_list)
all_cars_without_features.extend(all_cars_without_features)
assert(len(all_cars_without_features._matrix) == 812)
try:
    all_cars_without_features[float]
    raise ValueError("float should not work as __getitem__ object for Dataset")
except Exception as e:
    assert(e.args[0] == "key of type <class 'type'> not implemented for indexing Datasets")

try:
    all_cars_without_features[[float]]
    raise ValueError("float should not work as __getitem__ object for Dataset")
except Exception as e:
    assert(e.args[0] == "key of type <class 'list'> with <class 'type'> elements not implemented for indexing Datasets")

try:
    covariance([1, 2], [1])
    raise ValueError("covariance should be able to compute on lists of different lengths")
except Exception as e:
    assert(e.args[0] == "vector_x and vector_y must be the same length to compute covariance.")
