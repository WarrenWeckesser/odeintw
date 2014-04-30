#!/usr/bin/env python

# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

from distutils.core import setup


_descr = ('Solve complex and matrix differential equations '
          'with scipy.integrate.odeint.')

setup(name='odeintw', 
      version='0.0.1',
      description=_descr,
      author='Warren Weckesser',
      url='https://github.com/WarrenWeckesser/odeintw',
      packages=['odeintw', 'odeintw.tests'],
)
