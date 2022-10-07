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


def get_longest_path(graph: nx.DiGraph, begin, end) -> list:
    return list(nx.shortest_simple_paths(graph, begin, end))[-1]


def get_paths_from_to(graph: nx.DiGraph, _from: list, _to: list) -> list:
    paths = []
    for origin in _from:
        for destination in _to:
            try:
                paths.append(get_longest_path(graph, origin, destination))
            except nx.exception.NetworkXNoPath:
                continue
    return paths


def get_distance_by_nodes(graph: nx.DiGraph) -> Dict:
    longest_path = nx.dag_longest_path(graph)
    end_of_path = longest_path[-1]

    distances = {}

    for node in longest_path:
        distances[node] = longest_path.index(node)

    for node in [n for n in graph.nodes if n not in longest_path]:
        try:
            path = get_longest_path(graph, node, end_of_path)
            distances[node] = len(longest_path) - len(path)
        except nx.exception.NetworkXNoPath:
            pass

        if node not in distances:
            for path in get_paths_from_to(graph, list(graph.nodes), [node]):
                distance_via_current_node = distances[path[0]] + len(path) - 1
                if node not in distances or distances[node] < distance_via_current_node:
                    distances[node] = distance_via_current_node

    return distances
