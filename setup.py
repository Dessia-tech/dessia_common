# -*- coding: utf-8 -*-
"""
@author: Steven Masfaraud
"""

from setuptools import setup
import re

def readme():
    with open('README.rst') as f:
        return f.read()

with open('dessia_common/__init__.py','r') as f:
    metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", f.read()))

#print(metadata)

#import powertransmission
setup(name='dessia_common',
      version=metadata['version'],
      description="Common tools for DessIA software",
      long_description='',
      keywords='',
      url='',
#      cmdclass['register']=None,
      author='Steven Masfaraud',
      author_email='masfaraud@dessia.tech',
      packages=['dessia_common'],
      install_requires=[''])

