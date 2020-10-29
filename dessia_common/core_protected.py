#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: language_level=3

"""

"""

import inspect
from copy import deepcopy

from typing import Union
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7
import dessia_common as dc
import dessia_common.typings as dt


