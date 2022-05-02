import numpy as np 

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from mr_scl.dynamics import equinoctial_averaged_derivatives, M_matrix, costates_estimates, u_vector
from mr_scl.constants import mu_EARTH, R_EARTH, g0
from mr_scl.coc import kep2eqnct, eqnct2kep_avg


# Characteristic time, length and mass ([s], [km], [kg])
t_ref = 86400.000
l_ref = 42241.096
m_ref =  2000.000

# Characteristic velocity and acceleration ([km/s], [km/s^2])
v_ref = l_ref / t_ref
a_ref = v_ref / t_ref

# Scaled constants
# ================

mu_scl = mu_EARTH * t_ref**2 / l_ref**3
g0_scl = g0 * t_ref**2 / l_ref


# Orbit characteristics
SMA  = ( 20000 + R_EARTH ) / l_ref
ECC  = 0.3
INC  = 1.2
AOP  = 0
RAAN = 0 
TA   = np.linspace(0, 2*np.pi, 100)

# S/C characteristics
mass = 2e3 / m_ref
Isp = 2000 / t_ref

T_max = 2e-3 / m_ref / a_ref
delta = 1.

# Costates
costates = np.array([1, 0, 0, 1, 0])


keplerian_coord = np.array([SMA, ECC, INC, AOP, RAAN, TA])
equinoctial_coord = kep2eqnct(np.array([SMA, ECC, INC, AOP, RAAN, TA]))


    
# t_span = (0, ( 10 * 86400 ) / t_ref)

# propagate = solve_ivp(fun=equinoctial_averaged_derivatives, t_span=t_span, y0=np.concatenate((equinoctial_coord[:-1], [mass])), \
#     method='RK45', args=(costates, Isp, T_max, delta, mu_scl, g0_scl), atol=1e-8, rtol=1e-8)

# keplerian_results = np.ndarray(shape=(5, propagate.y.shape[1]))

# for k, t in enumerate(propagate.t):
#     keplerian_results[:, k] = eqnct2kep_avg(propagate.y[:, k])


# KEP_PRM = ['SMA', 'ECC', 'INC', 'AOP', 'RAAN']
# EQN_PRM = ['p', 'f', 'g', 'h', 'k']

# fig = plt.figure(figsize=(10, 7))
# ax_K = fig.add_subplot(211)

# for i in range(5):
#     ax_K.plot(propagate.t, keplerian_results[i], label='{}'.format(KEP_PRM[i]))

# plt.title('Keplerian parameters')
# plt.grid()
# plt.legend()

# ax_E = fig.add_subplot(212)

# for i in range(5):
#     ax_E.plot(propagate.t, propagate.y[i], label='{}'.format(EQN_PRM[i]))

# plt.title('Equinoctial parameters')
# plt.grid()
# plt.legend()
# plt.show()

