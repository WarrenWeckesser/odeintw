# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_, dec, run_module_suite)
import scipy
from scipy.integrate import odeint
from odeintw._odeintw import (_complex_to_real_jac, _transform_banded_jac,
                              odeintw)


_scipy_version = tuple(int(n) for n in scipy.__version__.split('.')[:3])
_banded_bug_msg = "known bug in scipy.integrate.odeint for scipy versions before 0.14.0"


def test_complex_to_real_jac():
    z = np.array([[1+2j]])
    r = _complex_to_real_jac(z)
    yield assert_array_equal, r, np.array([[1, -2], [2, 1]])
    z = np.array([[1+2j, 3+4j],
                  [5+6j, 7+8j]])
    r = _complex_to_real_jac(z)
    expected = np.array([[1, -2, 3, -4],
                         [2,  1, 4,  3],
                         [5, -6, 7, -8],
                         [6,  5, 8,  7]])
    yield assert_array_equal, r, expected


@dec.knownfailureif(_scipy_version < (0, 14, 0), _banded_bug_msg) 
def test_transform_banded_jac():
    j = np.array([[0,  0,  1,  2],
                  [0,  0,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12]])
    t = _transform_banded_jac(j)
    expected = np.array([[0,  0,  0,  2],
                         [0,  0,  1,  4],
                         [0,  6,  3,  8],
                         [5, 10,  7, 12],
                         [9,  0, 11,  0]])
    yield assert_array_equal, t, expected


def system1_complex(z, t):
    return z * (1 - z)

def system1_real(z, t):
    x, y = z
    return [x*(1-x) + y**2, y - 2*x*y]


def test_odeintw_complex():
    z0 = 0.25 + 0.5j
    t = np.linspace(0, 1, 5)
    zsol = odeintw(system1_complex, z0, t)
    zv = zsol.view(np.float64)
    sol = odeintw(system1_real, [z0.real, z0.imag], t)
    yield assert_allclose, zv, sol


def system2_array(a, t):
    return -a.T

def system2_vector(a, t):
    a11, a12, a21, a22 = a
    return [-a11, -a21, -a12, -a22]

def test_odeint_array():
    t = np.linspace(0, 1, 5)
    a0 = np.array([[1, -2], [3, -4]])
    asol = odeintw(system2_array, a0, t)
    avec = asol.reshape(-1, 4)
    sol = odeintw(system2_vector, a0.ravel(), t)
    yield assert_allclose, avec, sol


def system3_func(y, t, c):
    return c.dot(y)


def system3_funcz(y, t, c):
    # Same calculation as `system3_func`, but computed using real arrays,
    # so the calculation in `dot` should follow the same code path for
    # both the real and complex examples below.
    creal = _complex_to_real_jac(c)
    dydt = creal.dot(y.view(np.float64))
    return dydt.view(np.complex128)


def system3_jac(y, t, c):
    return c

def system3_bjac_cols(y, t, c):
    return np.column_stack( (np.r_[0, np.diag(c, 1)], np.diag(c)) )

def system3_bjac_rows(y, t, c):
    return np.row_stack( (np.r_[0, np.diag(c, 1)], np.diag(c)) )


def test_system3():
    c = np.array([[-20+1j, 5-1j,      0,       0], 
                  [     0, -0.1,  1+2.5j,      0],
                  [     0,    0,      -1,    0.5],
                  [     0,    0,       0,  -5+10j]])

    z0 = np.arange(1,5.0) + 0.5j

    t = np.linspace(0, 250, 11)

    common_kwargs = dict(full_output=True, atol=1e-12, rtol=1e-10,
                         mxstep=1000)

    sol0, info0 = odeintw(system3_funcz, z0, t, Dfun=system3_jac,
                          args=(c,), **common_kwargs)
    nje0 = info0['nje']

    x0 = z0.view(np.float64)
    sol1, info1 = odeint(system3_func, x0, t, Dfun=system3_jac,
                         args=(_complex_to_real_jac(c),), **common_kwargs)
    nje1 = info1['nje']

    # Using assert_array_equal here is risky.  The system definitions have
    # been defined so the call to odeint in odeintw follows the same
    # code path as the call to odeint above.  Still, floating point operations
    # aren't necessarily deterministic.
    yield assert_array_equal, sol0.view(np.float64), sol1
    yield assert_array_equal, nje0, nje1


@dec.knownfailureif(_scipy_version < (0, 14, 0), _banded_bug_msg) 
def test_system3_banded():
    c = np.array([[-20+1j, 5-1j,      0,       0], 
                  [     0, -0.1,  1+2.5j,      0],
                  [     0,    0,      -1,    0.5],
                  [     0,    0,       0,  -5+10j]])

    common_kwargs = dict(args=(c,), full_output=True, atol=1e-12, rtol=1e-10,
                         mxstep=1000)
    z0 = np.arange(1,5.0) + 0.5j
    t = np.linspace(0, 250, 11)

    sol0, info0 = odeintw(system3_func, z0, t, Dfun=system3_jac,
                          **common_kwargs)

    sol1, info1 = odeintw(system3_func, z0, t, Dfun=system3_bjac_cols,
                          ml=0, mu=1, col_deriv=True, **common_kwargs)
    sol2, info2 = odeintw(system3_func, z0, t, Dfun=system3_bjac_rows,
                          ml=0, mu=1, **common_kwargs)

    yield assert_allclose, sol0, sol1
    yield assert_allclose, sol0, sol2
    # The same code paths should have been followed in computing
    # sol1 and sol2, so the number of jacobian evaluations should
    # be the same.
    yield assert_array_equal, info1['nje'], info2['nje']


if __name__ == "__main__":
    run_module_suite()
