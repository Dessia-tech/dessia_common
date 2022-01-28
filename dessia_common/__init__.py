#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pkg_resources
from .core import *

__version__ = pkg_resources.require("dessia_common")[0].version

FLOAT_TOLERANCE = 1e-9
