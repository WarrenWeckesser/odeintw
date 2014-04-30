#!/usr/bin/env python

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
