#!/usr/bin/env python

# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

from os import path
from distutils.core import setup


def get_odeintw_version():
    """
    Find the value assigned to __version__ in odeintw/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in odeintw/__init__.py.  It returns the string version-string, or None if
    such a line is not found.
    """
    with open(path.join("odeintw", "__init__.py"), "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


_descr = ('Solve complex and matrix differential equations '
          'with scipy.integrate.odeint.')

setup(name='odeintw',
      version=get_odeintw_version(),
      description=_descr,
      author='Warren Weckesser',
      url='https://github.com/WarrenWeckesser/odeintw',
      packages=['odeintw', 'odeintw.tests'])
