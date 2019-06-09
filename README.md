odeintw
=======

`odeintw` provides a wrapper of `scipy.integrate.odeint` that allows it to
handle complex and matrix differential equations.  That is, it can solve
equations of the form

    dZ/dt = F(Z, t, param1, param2, ...)

where `t` is real and `Z` is a real or complex array.

Since `odeintw` is just a wrapper of `scipy.integrate.odeint`, it requires
`scipy` to be installed.

`odeintw` is available on PyPI: https://pypi.org/project/odeintw/


Example 1
---------

To solve the equations

    dz1/dt = -z1 * (K - z2)
    dz2/dt = L - M*z2

where `K`, `L` and `M` are (possibly complex) constants, we first define the
right-hand-side of the differential equations::

    def zfunc(z, t, K, L, M):
        z1, z2 = z
        return [-z1 * (K - z2), L - M*z2]

The Jacobian is

    def zjac(z, t, K, L, M):
        z1, z2 = z
        jac = np.array([[z2 - K, z1], [0, -M]])
        return jac

The following calls `odeintw` with appropriate arguments

    # Initial conditions.
    z0 = np.array([1+2j, 3+4j])

    # Desired time samples for the solution.
    t = np.linspace(0, 5, 101)

    # Parameters.
    K = 2
    L = 4 - 2j
    M = 2.5

    # Call odeintw
    z, infodict = odeintw(zfunc, z0, t, args=(K, L, M), Dfun=zjac,
                          full_output=True)

The components of the solution can be plotted with `matplotlib` as follows

    import matplotlib.pyplot as plt

    color1 = (0.5, 0.4, 0.3)
    color2 = (0.2, 0.2, 1.0)
    plt.plot(t, z[:, 0].real, color=color1, label='z1.real', linewidth=1.5)
    plt.plot(t, z[:, 0].imag, '--', color=color1, label='z1.imag', linewidth=2)
    plt.plot(t, z[:, 1].real, color=color2, label='z2.real', linewidth=1.5)
    plt.plot(t, z[:, 1].imag, '--', color=color2, label='z2.imag', linewidth=2)
    plt.xlabel('t')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

Plot:

![](https://github.com/WarrenWeckesser/odeintw/blob/master/examples/odeintw_example1.png)


Example 2
---------

We'll solve the matrix differential equation

    dA/dt = C * A

where `A` and `C` are real 2x2 matrices.

The differential equation is defined with the function

    def asys(a, t, c):
        return c.dot(a)

Both `a` and `c` are assumed to be `n x n` matrices.  The function
`asys` will work for any `n`, but we'll specialize to `2 x 2` in our
implementation of the Jacobian:

    def ajac(a, t, c):
        # asys returns [[F[0,0](a,t), F[0,1](a,t),
        #                F[1,0](a,t), F[1,1](a,t)]]
        # This function computes jac[m, n, i, j]
        # jac[m, n, i, j] holds dF[m,n]/da[i,j]
        jac = np.zeros((2,2,2,2))
        jac[0, 0, 0, 0] = c[0, 0]
        jac[0, 0, 1, 0] = c[0, 1]
        jac[0, 1, 0, 1] = c[0, 0]
        jac[0, 1, 1, 1] = c[0, 1]
        jac[1, 0, 0, 0] = c[1, 0]
        jac[1, 0, 1, 0] = c[1, 1]
        jac[1, 1, 0, 1] = c[1, 0]
        jac[1, 1, 1, 1] = c[1, 1]

(As with `odeint`, giving an explicit Jacobian is optional.)

Now create the arguments and call `odeintw`:

    # The matrix of coefficients `c`.  This is passed as an
    # extra argument to `asys` and `ajac`.
    c = np.array([[-0.5, -1.25],
                  [ 0.5, -0.25]])

    # Desired time samples for the solution.
    t = np.linspace(0, 10, 201)

    # a0 is the initial condition.
    a0 = np.array([[0.0, 1.0],
                   [2.0, 3.0]])

    # Call `odeintw`.
    sol = odeintw(asys, a0, t, Dfun=ajac, args=(c,))


The solution can be plotted with `matplotlib`:

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    color1 = (0.5, 0.4, 0.3)
    color2 = (0.2, 0.2, 1.0)
    plt.plot(t, sol[:, 0, 0], color=color1, label='a[0,0]')
    plt.plot(t, sol[:, 0, 1], color=color2, label='a[0,1]')
    plt.plot(t, sol[:, 1, 0], '--', color=color1, linewidth=1.5, label='a[1,0]')
    plt.plot(t, sol[:, 1, 1], '--', color=color2, linewidth=1.5, label='a[1,1]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

Plot:

![](https://github.com/WarrenWeckesser/odeintw/blob/master/examples/odeintw_example2.png)


*Copyright (c) 2015, Warren Weckesser*

All rights reserved.
See the LICENSE file for license information.
