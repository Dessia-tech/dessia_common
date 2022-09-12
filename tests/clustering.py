"""
Cluster.py package testing.
"""
import json
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_small, rand_data_large
from dessia_common.datatools import HeterogeneousList, CategorizedList

# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
# Auto-generated heterogeneous small dataset with nb_clusters clusters of points in nb_dims dimensions
small_RandDatas_heterogeneous = HeterogeneousList(rand_data_small)
# Auto-generated heterogeneous large dataset with nb_clusters clusters of points in nb_dims dimensions
big_RandDatas_heterogeneous = HeterogeneousList(rand_data_large)

# Build CategorizedLists
clustered_cars_without = CategorizedList.from_dbscan(all_cars_without_features, eps=40)
clustered_cars_with = CategorizedList.from_dbscan(all_cars_with_features, eps=40)
aggclustest_clustered = CategorizedList.from_agglomerative_clustering(big_RandDatas_heterogeneous, n_clusters=10)
kmeanstest_clustered = CategorizedList.from_kmeans(small_RandDatas_heterogeneous, n_clusters=10, scaling=True)

# Split lists into labelled lists
split_cars_without = clustered_cars_without.clustered_sublists()
split_cars_with = clustered_cars_with.clustered_sublists()
aggclustest_split = aggclustest_clustered.clustered_sublists()
kmeanstest_split = kmeanstest_clustered.clustered_sublists()

# Test print
clustered_cars_without.labels[0] = 15000
clustered_cars_without.labels[1] = -1
clustered_cars_without.labels[2:100] = [999999]*len(clustered_cars_without[2:100])
print(clustered_cars_without)
hlist = HeterogeneousList(all_cars_wi_feat, name="cars")
clist = CategorizedList.from_agglomerative_clustering(hlist, n_clusters=10, name="cars")
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
