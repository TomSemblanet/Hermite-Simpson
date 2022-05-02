import numpy as np 
import cppad_py
from scipy.integrate import trapezoid

from mr_scl.coc import kep2eqnct

def M_matrix(x, L, mu_scl):
    """ Returns the `M` matrix """

    p, f, g, h, k = x

    sin_L = np.sin(L)
    cos_L = np.cos(L)

    alpha = np.sqrt(p / mu_scl)
    q = 1 + f*cos_L + g*sin_L
    s2 = 1 + h*h + k*k

    M = alpha * np.array([[      0,               2 * p / q,                         0 ], 
                          [  sin_L,   ((q+1)*cos_L + f) / q, -g * (h*sin_L - k*cos_L)/q],
                          [ -cos_L,   ((q+1)*sin_L + g) / q,  f * (h*sin_L - k*cos_L)/q],
                          [      0,                       0,             s2/2/q * cos_L],
                          [      0,                       0,             s2/2/q * sin_L]])

    return M


def F_vector(M, m, u, delta, T_max, Isp, g0_scl):
    """ Return the F(x, m; L, u, delta) vector """ 

    x_dot =   delta * T_max / m * np.dot( u, np.transpose(M) )
    m_dot = - delta * T_max / g0_scl / Isp

    return np.concatenate((x_dot, [m_dot]))


def u_vector(M, costates):
    """ Returns the Lawden's primer vector as a function of the `M` matrix and the costates"""

    dot_prd = np.dot(costates, M)

    return dot_prd / np.linalg.norm(dot_prd)


def dt_dL_ratio(x_bar, m_bar, L, u, T_max, mu_scl):
    """ Returns the dt_dL ratio """

    p_bar, f_bar, g_bar, h_bar, k_bar = x_bar

    alpha = np.sqrt(p_bar / mu_scl)
    cos_L = np.cos(L)
    sin_L = np.sin(L)
    q = 1 + f_bar*cos_L + g_bar*sin_L

    dt_dL = 1 / (np.sqrt(mu_scl * p_bar) * (q/p_bar)**2 + alpha*( (h_bar*sin_L - k_bar*cos_L) / q *\
     (T_max / m_bar * u[2]) ))

    return dt_dL

def inner_term(L, y_bar, costates, T_max, Isp, delta, mu_scl, g0_scl):
    """ Computation of the integral inner term :  F(x, m, L, u, 1) * (dt/dL) * delta. """

    # Extraction of the vector
    x_bar = y_bar[:-1]
    m_bar = y_bar[-1]

    # Computation of the `M` matrix
    M = M_matrix(x_bar, L, mu_scl)
    
    # Computation of the primer vector as a function of the `M` matrix and the costates
    u = u_vector(M, costates)

    # Computation of the derivatives vector
    F = F_vector(M, m_bar, u, 1, T_max, Isp, g0_scl)

    # Computation of the ratio dt to dL
    dt_dL = dt_dL_ratio(x_bar, m_bar, L, u, T_max, mu_scl)

    return delta * F * dt_dL 


def equinoctial_averaged_derivatives(t, y_bar, costates, Isp, T_max, delta, mu_scl, g0_scl, N=50):
    """ Computation of the averaged states derivatives using `Trapezoidal` rule for integration.

        Parameters
        ----------
        t : float
            Time parameter (used by the `solve_ivp` method to propagate the EOMs)
        y_bar : array
            Averaged equinoctial parameters (except the True Longitude `L`) (+) S/C mass
        costates : array
            Costates of the equinoctial parameters (except the True Longitude `L`)
        Isp : float
            S/C Specific Impulse
        T_max : float
            S/C maximum thrust
        delta : float
            Throttle level (= 1 for minimum time problem)
        mu_scl : float
            Scaled characteristic parameter 
        g0_scl : float    
            Earth gravitation at sea level scaled 

        Returns
        -------
        y_bar_dot : array
           Averaged equinoctial parameters derivatives (expect the True Longitude `L`) (+) S/C mass derivative
                

    """
    # Extraction of the equinoctial averaged parameters (+) S/C mass
    p_bar, f_bar, g_bar, h_bar, k_bar, m_bar = y_bar

    # Calculation of the orbital period
    T = 2*np.pi * np.sqrt( (p_bar/(1 - f_bar**2 - g_bar**2))**3 / mu_scl)


    # Calculation of the integrale inner term before integration
    Ls = np.linspace(0, 2*np.pi, N)
    y = np.ndarray(shape=(6, N), dtype=cppad_py.a_double)

    for k, L in enumerate(Ls):
        y[:, k] = inner_term(L=L, y_bar=y_bar, \
                    costates=costates, T_max=T_max, Isp=Isp, delta=delta, mu_scl=mu_scl, g0_scl=g0_scl)

    # Integration over the interval [0, 2*pi]
    y_bar_dot = 1/T * trapezoid(y=y, x=Ls, dx=(Ls[1]-Ls[0]))


    return y_bar_dot


if __name__ == '__main__':

    y_bar = np.random.rand(1,6)[0]
    costates = np.random.rand(1,5)[0]
    Isp = 0.03587962962962963
    T_max = 0.08836134365453019
    delta = 1.0
    mu_scl = 39.47841669104313
    g0_scl = 1733.649562501882


    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for N in np.linspace(20, 1000):
        result = equinoctial_averaged_derivatives(0, y_bar, costates, Isp, T_max, delta, mu_scl, g0_scl, int(N))
        ax.plot(N, result[0], 'o', color='blue')

    plt.grid()
    plt.show()

    

