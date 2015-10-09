# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

import numpy as np
import matplotlib.pyplot as plt

from odeintw import odeintw


# Matrix differential equation
#     da/dt = c*a
# where a and c are (for example) 2x2 matrices.

def asys(a, t, c):
    return c.dot(a)


def ajac(a, t, c):
    # asys returns [[F[0,0](a,t), F[0,1](a,t),
    #                F[1,0](a,t), F[1,1](a,t)]]
    # This function computes jac[m, n, i, j]
    # jac[m, n, i, j] holds dF[m,n]/da[i,j]
    jac = np.zeros((2, 2, 2, 2))
    jac[0, 0, 0, 0] = c[0, 0]
    jac[0, 0, 1, 0] = c[0, 1]
    jac[0, 1, 0, 1] = c[0, 0]
    jac[0, 1, 1, 1] = c[0, 1]
    jac[1, 0, 0, 0] = c[1, 0]
    jac[1, 0, 1, 0] = c[1, 1]
    jac[1, 1, 0, 1] = c[1, 0]
    jac[1, 1, 1, 1] = c[1, 1]

    return jac

c = np.array([[-0.5, -1.25],
              [ 0.5, -0.25]])
t = np.linspace(0, 10, 201)

# a0 is the initial condition.
a0 = np.array([[0.0, 1.0],
               [2.0, 3.0]])

sol = odeintw(asys, a0, t, Dfun=ajac, args=(c,))

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
