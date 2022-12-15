#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:28:55 2021

@author: masfaraud
"""

import dessia_common.displays as dcd
import networkx as nx

G = nx.karate_club_graph()

dcd.draw_networkx_graph(G)
