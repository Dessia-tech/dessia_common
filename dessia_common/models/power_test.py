#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:19:08 2021

@author: masfaraud
"""

import dessia_common.tests

components1 = [dessia_common.tests.Component(0.7),
               dessia_common.tests.Component(0.9),
               dessia_common.tests.Component(0.95)]

components2 = [dessia_common.tests.Component(0.7),
               dessia_common.tests.Component(0.96),
               dessia_common.tests.Component(0.95),
               dessia_common.tests.Component(0.99)]


component_connections1 = [dessia_common.tests.ComponentConnection(components1[0], components1[1]),
                          dessia_common.tests.ComponentConnection(components1[1], components1[2])
                          ]

component_connections2 = [dessia_common.tests.ComponentConnection(components2[0], components2[1]),
                          dessia_common.tests.ComponentConnection(components2[1], components2[2]),
                          dessia_common.tests.ComponentConnection(components2[2], components2[3])
                          ]


system1 = dessia_common.tests.System(components1, component_connections1, 'system1')  # {components1[1]: 0.32})
system2 = dessia_common.tests.System(components2, component_connections2, 'system2')  # {components2[0]: 0.32})

usage1 = dessia_common.tests.SystemUsage([0, 1., 2., 5., 8.], [0., 100., 150, 350, 120.])
usage2 = dessia_common.tests.SystemUsage([0, 1., 2., 5., 10.], [0., 76., 150, 357, 110.])


simulation1 = system1.power_simulation(usage1)
simulation2 = system2.power_simulation(usage1)
simulation3 = system1.power_simulation(usage2)
simulation4 = system2.power_simulation(usage2)


simulation_list = dessia_common.tests.SystemSimulationList([simulation1, simulation2, simulation3, simulation4])
