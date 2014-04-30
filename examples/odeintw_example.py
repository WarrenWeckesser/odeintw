# Copyright (c) 2014, Warren Weckesser
# All rights reserved.
# See the LICENSE file for license information.

import numpy as np
import matplotlib.pyplot as plt

from odeintw import odeintw


# Generate a solution to:
#     dz1/dt = -z1 * (K - z2)
#     dz2/dt = L - M*z2
# K, L and M are fixed parameters.  z1(t) and z2(t) are complex-
# valued functions of t.

# Define the right-hand-side of the differential equation.
def zfunc(z, t, K, L, M):
    z1, z2 = z
    return [-z1 * (K - z2), L - M*z2]

def zjac(z, t, K, L, M):
    z1, z2 = z
    jac = np.array([[z2 - K, z1], [0, -M]])
    return jac

# Set up the inputs and call odeintw to solve the system.
z0 = np.array([1+2j, 3+4j])
t = np.linspace(0, 5, 101)
K = 2
L = 4 - 2j
M = 2.5
z, infodict = odeintw(zfunc, z0, t, args=(K,L,M), Dfun=zjac, full_output=True)

plt.figure(2)
plt.clf()
plt.plot(t, z[:,0].real, label='z1.real')
plt.plot(t, z[:,0].imag, label='z1.imag')
plt.plot(t, z[:,1].real, label='z2.real')
plt.plot(t, z[:,1].imag, label='z2.imag')
plt.xlabel('t')
plt.grid(True)
plt.legend(loc='best')

plt.show()
