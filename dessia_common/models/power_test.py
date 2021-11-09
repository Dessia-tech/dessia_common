#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:19:08 2021

@author: masfaraud
"""

import dessia_common.tests

components = [dessia_common.tests.Component(0.7),
              dessia_common.tests.Component(0.9),
              dessia_common.tests.Component(0.95)]

component_connections = [dessia_common.tests.ComponentConnection(components[0], components[1]),
                         dessia_common.tests.ComponentConnection(components[1], components[2])
                         ]

system = dessia_common.tests.System(components, component_connections)

usage = dessia_common.tests.SystemUsage([0, 1., 2., 5., 8.], [0., 100., 150, 350, 120.])

simulation = system.power_simulation(usage)

