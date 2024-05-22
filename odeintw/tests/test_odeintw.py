# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy
from scipy.integrate import odeint
from odeintw._odeintw import (_complex_to_real_jac, _transform_banded_jac,
                              odeintw)


_scipy_version = tuple(int(n) for n in scipy.__version__.split('.')[:3])
_banded_bug_msg = ("known bug in scipy.integrate.odeint for scipy versions "
                   "before 0.14.0")


C = np.array([[-20+1j, 5-1j,      0,       0],
              [     0, -0.1,  1+2.5j,      0],
              [     0,    0,      -1,    0.5],
              [     0,    0,       0,  -5+10j]])


def test_complex_to_real_jac():
    z = np.array([[1+2j]])
    r = _complex_to_real_jac(z)
    assert_array_equal(r, np.array([[1, -2], [2, 1]]))
    z = np.array([[1+2j, 3+4j],
                  [5+6j, 7+8j]])
    r = _complex_to_real_jac(z)
    expected = np.array([[1, -2, 3, -4],
                         [2,  1, 4,  3],
                         [5, -6, 7, -8],
                         [6,  5, 8,  7]])
    assert_array_equal(r, expected)


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
    assert_array_equal(t, expected)


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
    assert_allclose(zv, sol)


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
    assert_allclose(avec, sol)


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
    return np.column_stack((np.r_[0, np.diag(c, 1)], np.diag(c)))


def system3_bjac_rows(y, t, c):
    return np.vstack((np.r_[0, np.diag(c, 1)], np.diag(c)))


def test_system3():
    z0 = np.arange(1, 5.0) + 0.5j

    t = np.linspace(0, 250, 11)

    common_kwargs = dict(full_output=True, atol=1e-12, rtol=1e-10,
                         mxstep=1000)

    sol0, info0 = odeintw(system3_funcz, z0, t, Dfun=system3_jac,
                          args=(C,), **common_kwargs)
    nje0 = info0['nje']

    x0 = z0.view(np.float64)
    sol1, info1 = odeint(system3_func, x0, t, Dfun=system3_jac,
                         args=(_complex_to_real_jac(C),), **common_kwargs)
    nje1 = info1['nje']

    # Using assert_array_equal here is risky.  The system definitions have
    # been defined so the call to odeint in odeintw follows the same
    # code path as the call to odeint above.  Still, floating point operations
    # aren't necessarily deterministic.
    assert_array_equal(sol0.view(np.float64), sol1)
    assert_array_equal(nje0, nje1)


def test_system3_banded():
    c = np.array([[-20+1j, 5-1j,      0,       0],
                  [     0, -0.1,  1+2.5j,      0],
                  [     0,    0,      -1,    0.5],
                  [     0,    0,       0,  -5+10j]])

    common_kwargs = dict(args=(c,), full_output=True, atol=1e-12, rtol=1e-10,
                         mxstep=1000)
    z0 = np.arange(1, 5.0) + 0.5j
    t = np.linspace(0, 250, 11)

    sol0, info0 = odeintw(system3_func, z0, t, Dfun=system3_jac,
                          **common_kwargs)

    sol1, info1 = odeintw(system3_func, z0, t, Dfun=system3_bjac_cols,
                          ml=0, mu=1, col_deriv=True, **common_kwargs)
    sol2, info2 = odeintw(system3_func, z0, t, Dfun=system3_bjac_rows,
                          ml=0, mu=1, **common_kwargs)

    assert_allclose(sol0, sol1)
    assert_allclose(sol0, sol2)
    # The same code paths should have been followed in computing
    # sol1 and sol2, so the number of jacobian evaluations should
    # be the same.
    assert_array_equal(info1['nje'], info2['nje'])


def system1_complex_tfirst(t, z):
    return z * (1 - z)


def test_tfirst_system1():
    z0 = 0.25 + 0.5j
    t = np.linspace(0, 1, 5)
    sol_zfirst = odeintw(system1_complex, z0, t)
    sol_tfirst = odeintw(system1_complex_tfirst, z0, t, tfirst=True)
    assert_allclose(sol_tfirst, sol_zfirst)


def system3_func_tfirst(t, y, c):
    return c.dot(y)


def system3_jac_tfirst(t, y, c):
    return c


def test_system3_tfirst():
    z0 = np.array([1.0, 2+3j, -4j, 5])
    t = np.linspace(0, 250, 11)
    common_kwargs = dict(atol=1e-12, rtol=1e-10, mxstep=1000)

    sol = odeintw(system3_func, z0, t, Dfun=system3_jac,
                  args=(C,), **common_kwargs)
    sol_tfirst = odeintw(system3_func_tfirst, z0, t, Dfun=system3_jac_tfirst,
                         args=(C,), tfirst=True, **common_kwargs)
    assert_allclose(sol, sol_tfirst)


def test_complex_simple_scalar_integration():
    # The exact solution is z0 + k*t**2

    def sys(z, t, k):
        return 2*k*t

    def sys_tfirst(t, z, k):
        return 2*k*t

    k = 3
    z0 = 1+2j,
    t = np.array([0, 0.5, 1])
    sol = odeintw(sys, z0, t, args=(k,))
    assert_allclose(sol, z0 + k*t.reshape(-1, 1)**2)

    sol = odeintw(sys_tfirst, z0, t, args=(k,), tfirst=True)
    assert_allclose(sol, z0 + k*t.reshape(-1, 1)**2)


def test_matrix_y0():
    # Regression test for gh-8.

    def sys(x, t):
        return -x

    # The numpy.matrix class is deprecated, so this use of matrix will generate
    # a warning with recent versions of NumPy.  When the matrix class is
    # removed from numpy, this entire test can be removed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        y0 = np.matrix([[0, 2], [3, 1]])
        t = np.array([0, 0.5, 1])
        sol = odeintw(sys, y0, t, rtol=1e-12)
        assert_allclose(sol, np.asarray(y0)*np.exp(-t.reshape((len(t), 1, 1))))
