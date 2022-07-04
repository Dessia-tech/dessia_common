"""
Cluster.py package testing.
"""
import json
import pkg_resources
from dessia_common import tests, cluster
import dessia_common.workflow as wf


csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars = tests.Car.from_csv(csv_cars)

dbtest = cluster.ClusterResult.from_dbscan(all_cars, eps=100)
aggclustest = cluster.ClusterResult.from_agglomerative_clustering(all_cars, n_clusters=5)
kmeanstest = cluster.ClusterResult.from_kmeans(all_cars, n_clusters=5)


db_list = dbtest.data_to_clusters(all_cars, dbtest.labels)
agg_list = aggclustest.data_to_clusters(all_cars, aggclustest.labels)
kmeans_list = kmeanstest.data_to_clusters(all_cars, kmeanstest.labels)

# dbtest.check_dimensionality()
# aggclustest.check_dimensionality()
# kmeanstest.check_dimensionality()

dbtest.plot()
# aggclustest.plot()
# kmeanstest.plot()

# dbtest._check_platform()
# aggclustest._check_platform()
# kmeanstest._check_platform()

# data_method = wf.MethodType(class_=tests.Car, name='from_csv')
# block_data = wf.ClassMethod(method_type=data_method, name='data load')

# cluster_method = wf.MethodType(class_=cluster.ClusterResult, name='from_agglomerative_clustering')
# block_cluster = wf.ClassMethod(method_type=cluster_method, name='clustering')

# display_cluster = wf.Display(name='Display Cluster')

# block_workflow = [block_cluster, display_cluster]
# pipe_worflow = [wf.Pipe(block_cluster.outputs[0], display_cluster.inputs[0])]
# workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

# workflow_run = workflow.run({workflow.index(block_cluster.inputs[0]): all_cars})
# cresult = workflow_run.output_value._display_from_selector('plot_data')

# # ff=workflow_run.output_value.display_settings()[1]
# # ff.selector

# workflow.plot()
# workflow.display_settings()
# workflow_run.output_value.plot()

# gg = workflow_run._display_from_selector('plot_data')
# json.dumps(gg.to_dict())
# workflow_run._displays
