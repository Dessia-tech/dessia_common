#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module for docstring parsing to platform and Sphinx auto documentation. """


from typing import Dict, Any, Tuple, get_type_hints
from dessia_common.utils.types import serialize_typing

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7
