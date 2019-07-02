#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""

import math
import dessia_common.workflow as workflow

def sinus_f(x: float) -> float:
    return math.sin(x)

def asin_f(x: float) -> float:
    return math.asin(x)


sinus = workflow.Function(sinus_f)
arcsin = workflow.Function(asin_f)

pipe1 = workflow.Pipe(sinus.outputs[0], arcsin.inputs[0])

workflow = workflow.WorkFlow([sinus, arcsin], [pipe1])

workflow.plot_graph()

workflow_run = workflow.run([math.pi/3])