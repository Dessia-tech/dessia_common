#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dessia_common.displays as dcd
import dessia_common.graph as dcg
import networkx as nx

G = nx.gn_graph(80)

dcd.draw_networkx_graph(G)

reg = dcg.extract_region(G, [1, 2])

assert 1 in reg

order = dcg.explore_tree_from_leaves(graph=G)
g = G.to_undirected()
connected_components = dcg.find_connected_components(graph=g)
minimum_spanning_tree = dcg.find_minimum_spanning_tree(graph=g)
graph2 = dcg.cut_tree_final_branches(reg)
