"""
Cluster.py package testing.
"""
import pkg_resources
from dessia_common import tests, cluster

csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars = tests.Car.from_csv(csv_cars)

dbtest = cluster.ClusterResult.from_dbscan(all_cars, eps=0.1)
aggclustest = cluster.ClusterResult.from_agglomerative_clustering(all_cars, n_clusters=5)
kmeanstest = cluster.ClusterResult.from_kmeans(all_cars, n_clusters=5)


db_list = dbtest.data_to_clusters(all_cars, dbtest.labels)
agg_list = aggclustest.data_to_clusters(all_cars, aggclustest.labels)
# kmeans_list = kmeanstest.data_to_clusters(all_cars, kmeanstest.labels)

# dbtest.check_dimensionality()
# aggclustest.check_dimensionality()
# # kmeanstest.check_dimensionality(all_cars)

# dbtest.plot_data()
# aggclustest.plot_data()
# # kmeanstest.plot_data()

dbtest._check_platform()
aggclustest._check_platform()
kmeanstest._check_platform()


