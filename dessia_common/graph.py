#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph helpers
"""
from typing import Dict

import networkx as nx


def explore_tree_from_leaves(graph):
    if not nx.is_directed_acyclic_graph(graph):
        raise NotImplementedError('Cycles in jsonpointers not handled')

    return list(nx.topological_sort(graph.reverse()))


def cut_tree_final_branches(graph: nx.DiGraph):

    number_node_removed = 1
    while number_node_removed:
        nodes_to_delete = []
        # number_node_removed = 0
        for node in graph.nodes():
            if graph.out_degree(node) == 0:
                # It is a leaf
                degree_in = graph.in_degree(node)
                current_node = node
                while degree_in == 1:
                    nodes_to_delete.append(current_node)
                    current_node = graph.predecessors(current_node)
                    degree_in = graph.in_degree(current_node)
        number_node_removed = len(nodes_to_delete)
        if number_node_removed:
            new_nodes = [n for n in graph.nodes if n not in nodes_to_delete]
            graph = nx.subgraph(graph, new_nodes)

    return graph


def extract_region(networkx_graph: nx.Graph, nodes, distance: int = 5):
    """
    Extract all nodes that are at distance from node
    """

    region_nodes = nodes[:]
    for node in nodes:
        if node not in networkx_graph:
            raise ValueError(f'node {node} is not in graph')
        succs = list(nx.dfs_successors(networkx_graph, source=node, depth_limit=distance))
        preds = list(nx.dfs_predecessors(networkx_graph, source=node, depth_limit=distance))
        ancs = list(nx.ancestors(networkx_graph, source=node))
        print(f'found {len(succs)} successors and {len(preds)} predecesors {len(ancs)} ancestors')
        region_nodes.extend(succs)
        region_nodes.extend(preds)
        region_nodes.extend(ancs)

    return networkx_graph.subgraph(region_nodes)


def get_distance_by_nodes(graph) -> Dict:
    longest_path = nx.dag_longest_path(graph)
    end_of_path = longest_path[-1]

    distances = {}
    untreated_nodes = [n for n in graph.nodes if n not in longest_path]

    for node in longest_path:
        distances[node] = longest_path.index(node)

    for node in untreated_nodes:
        # looking for nodes heading to end_of_path
        try:
            path = list(nx.shortest_simple_paths(graph, node, end_of_path))[-1]
            distances[node] = len(longest_path) - len(path)
        except nx.exception.NetworkXNoPath:
            continue

    for current_node in untreated_nodes:
        # looking for node that heads to current_node (which does not head to end_of_path)
        treated_nodes = [node for node in graph.nodes if node not in untreated_nodes]
        for treated_node in treated_nodes:
            try:
                paths = nx.shortest_simple_paths(graph, treated_node, current_node)
                longest_path = list(paths)[-1]
                distance_via_current_node = distances[treated_node] + len(longest_path) - 1
                if current_node not in distances or distances[current_node] < distance_via_current_node:
                    distances[current_node] = distance_via_current_node
            except nx.exception.NetworkXNoPath:
                continue

        if current_node not in distances:
            distances[current_node] = len(longest_path)

    return distances

