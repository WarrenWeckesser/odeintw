# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

from __future__ import print_function

import numpy as np
from scipy.integrate import odeint

from odeintw import odeintw
from odeintw._odeintw import _complex_to_real_jac


def func(y, t, c):
    return c.dot(y)


def funcz(y, t, c):
    # Same calculation as `func`, but computed using real arrays,
    # so the calculation in `dot` should follow the same code path
    # for both the real and complex examples below.
    creal = _complex_to_real_jac(c)
    dydt = creal.dot(y.view(np.float64))
    return dydt.view(np.complex128)

def jac(y, t, c):
    return c

def bjac_cols(y, t, c):
    return np.column_stack( (np.r_[0, np.diag(c, 1)], np.diag(c)) )

def bjac_rows(y, t, c):
    return np.row_stack( (np.r_[0, np.diag(c, 1)], np.diag(c)) )


c = np.array([[-20+1j, 5-1j,      0,       0], 
              [     0, -0.1,  1+2.5j,      0],
              [     0,    0,      -1,    0.5],
              [     0,    0,       0,  -5+10j]])
print(c)
print()

z0 = np.arange(1,5.0) + 0.5j

t = np.linspace(0, 250, 11)

common_kwargs = dict(args=(c,), full_output=True, atol=1e-12, rtol=1e-10, mxstep=1000)

sol0, info0 = odeintw(funcz, z0, t, Dfun=jac, **common_kwargs)
print(info0['nje'])

rargs = common_kwargs.copy()
rargs.pop('args')

x0 = z0.view(np.float64)
solr, infor = odeint(func, x0, t, Dfun=jac, args=(_complex_to_real_jac(c),), **rargs)
print(infor['nje'])

print("-----")

solbnj, infobnj = odeintw(func, z0, t, ml=0, mu=1, **common_kwargs)
print(infobnj['nje'])

sol2, info2 = odeint(func, x0, t, ml=1, mu=3, args=(_complex_to_real_jac(c),), **rargs)
print(info2['nje'])

print("-----")

sol1, info1 = odeintw(func, z0, t, Dfun=bjac_cols, ml=0, mu=1, col_deriv=True, **common_kwargs)
print(info1['nje'])

sol2, info2 = odeintw(func, z0, t, Dfun=bjac_rows, ml=0, mu=1, **common_kwargs)
print(info2['nje'])
