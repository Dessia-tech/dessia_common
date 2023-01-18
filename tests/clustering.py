"""
Cluster.py package testing.
"""
import json
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_small, rand_data_large
from dessia_common.datatools import HeterogeneousList, CategorizedList
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.cluster import ClusteredDataset

# When attribute _features is not specified in class Car
all_cars_without_features = Dataset(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = Dataset(all_cars_wi_feat)
# Auto-generated heterogeneous small dataset with nb_clusters clusters of points in nb_dims dimensions
small_RandDatas_heterogeneous = Dataset(rand_data_small)
# Auto-generated heterogeneous large dataset with nb_clusters clusters of points in nb_dims dimensions
big_RandDatas_heterogeneous = Dataset(rand_data_large)

# Build ClusteredDatasets
clustered_cars_without = ClusteredDataset.from_dbscan(all_cars_without_features, eps=40)
clustered_cars_with = ClusteredDataset.from_dbscan(all_cars_with_features, eps=40)
aggclustest_clustered = ClusteredDataset.from_agglomerative_clustering(big_RandDatas_heterogeneous, n_clusters=10)
kmeanstest_clustered = ClusteredDataset.from_kmeans(small_RandDatas_heterogeneous, n_clusters=10, scaling=True)

clustered_cars_with_list = ClusteredDataset.list_dbscan(all_cars_no_feat, eps=40)
aggclustest_clustered_list = ClusteredDataset.list_agglomerative_clustering(rand_data_large, n_clusters=10)
kmeanstest_clustered_list = ClusteredDataset.list_kmeans(rand_data_small, n_clusters=10, scaling=True)
assert(clustered_cars_with_list == clustered_cars_without)
assert(aggclustest_clustered_list == aggclustest_clustered)
assert(kmeanstest_clustered_list.n_clusters == 10)


# Split lists into labelled lists
split_cars_without = clustered_cars_without.clustered_sublists()
split_cars_with = clustered_cars_with.clustered_sublists()
aggclustest_split = aggclustest_clustered.clustered_sublists()
kmeanstest_split = kmeanstest_clustered.clustered_sublists()

# Centroids
assert(clustered_cars_with.cluster_real_centroids('mahalanobis')[0].to_vector()[1] == 0.135)
assert(clustered_cars_with.cluster_real_centroids('minkowski')[0].to_vector()[1] == 0.156)

# Test print
clustered_cars_without.labels[0] = 15000
clustered_cars_without.labels[1] = -1
clustered_cars_without.labels[2:100] = [999999] * len(clustered_cars_without[2:100])
print(clustered_cars_without)
hlist = Dataset(all_cars_wi_feat, name="cars")
clist = ClusteredDataset.from_agglomerative_clustering(hlist, n_clusters=10, name="cars")
split_clist = clist.clustered_sublists()
split_clist[0].name = "15g6e4rg84reh56rt4h56j458hrt56gb41rth674r68jr6"
print(split_clist)

# Test ClusterResults instances on platform
clustered_cars_without._check_platform()
clustered_cars_with._check_platform()
aggclustest_clustered._check_platform()
kmeanstest_clustered._check_platform()

# Test plots outside platform
cluscars_plot_data = clustered_cars_without.plot_data()
cluscars_plot_data = split_cars_with.plot_data()
# assert(json.dumps(cluscars_plot_data[0].to_dict())[42500:42550] == ' 2515.0, "acceleration": 14.8, "model": 78.0, "Clu')
# assert(json.dumps(cluscars_plot_data[1].to_dict())[52500:52550] == '4.3, "model": 80.0, "Cluster Label": -1}, {"mpg": ')
# assert(json.dumps(cluscars_plot_data[2].to_dict())[50:100] == 'te_names": ["Index of reduced basis vector", "Sing')


# =============================================================================
# JSON TESTS
# =============================================================================
dict_cars_without = clustered_cars_without.to_dict(use_pointers=True)
dict_cars_with = clustered_cars_with.to_dict(use_pointers=True)
dict_aggclustest = aggclustest_clustered.to_dict(use_pointers=True)
dict_kmeanstest = kmeanstest_clustered.to_dict(use_pointers=True)

# Cars without features
json_dict = json.dumps(dict_cars_without)
decoded_json = json.loads(json_dict)
deserialized_object = clustered_cars_without.dict_to_object(decoded_json)

# Cars with features
json_dict = json.dumps(dict_cars_with)
decoded_json = json.loads(json_dict)
deserialized_object = clustered_cars_with.dict_to_object(decoded_json)

# Small dataset
json_dict = json.dumps(dict_aggclustest)
decoded_json = json.loads(json_dict)
deserialized_object = aggclustest_clustered.dict_to_object(decoded_json)

# Large dataset
json_dict = json.dumps(dict_kmeanstest)
decoded_json = json.loads(json_dict)
deserialized_object = kmeanstest_clustered.dict_to_object(decoded_json)

# Missing tests after coverage report
try:
    clustered_cars_without + clustered_cars_without
    raise ValueError("ClusteredDataset should be summable")
except Exception as e:
    assert(e.args[0] == "Addition only defined for Dataset. A specific __add__ method is required for " +
           "<class 'dessia_common.datatools.cluster.ClusteredDataset'>")

# Exports XLS
clustered_cars_without.to_xlsx('clus_xls.xlsx')
split_cars_with.to_xlsx('clus_xls_2.xlsx')

# Retrocompatibility
Hlist = HeterogeneousList(all_cars_no_feat)
Clist = CategorizedList(all_cars_no_feat)
