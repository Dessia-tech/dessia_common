#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .core import *
import pkg_resources

__version__ = pkg_resources.require("dessia_common")[0].version

FLOAT_TOLERANCE = 1e-9
