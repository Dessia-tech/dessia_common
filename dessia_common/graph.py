#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:16:30 2021

@author: steven
"""

import networkx as nx



def explore_tree_from_leaves(graph:nx.DiGraph):
    # graph = graph.copy()
    exploration_order = []
    explored = {n:False for n in graph.nodes}
    nn = graph.number_of_nodes()

    while nn:
        found_node = False
        # Finding a starting node
        for node in graph.nodes:
            if not explored[node]:
                neighbors_explored = True
                for out_node in graph.successors(node):
                    if not explored[out_node]:
                        neighbors_explored = False
                        break
                if neighbors_explored:
                    # Mark explored
                    explored[node] = True
                    exploration_order.append(node)
                    nn -= 1
                    found_node = True
                    break
        if not found_node:
            raise ValueError('Can not find a node')
    return exploration_order