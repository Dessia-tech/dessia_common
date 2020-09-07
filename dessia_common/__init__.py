#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .core import * 
from .core import _open_source
import pkg_resources

__version__ = pkg_resources.require("dessia_common")[0].version
