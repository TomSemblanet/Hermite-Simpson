import numpy as np

import mr_scl.constants


def kep2cart(coe):
    """ Converts coordinates of a body from orbital elements (coe) into cartesian coordinates 
    Arguments : 
    
        coe (array) : Orbital elements of the body (SMA, ECC, INC, AOP, RAAN, TA)

    Returns : 
        r (array) : Concatenation of the body's position and velocity (X, Y, Z, VX, VY, VZ) in the geocentric frame
    """

    a, e, i, w, RA, TA = coe

    h = np.sqrt(constants.mu_EARTH * a * (1 - e**2))

    rp = (h**2/constants.mu_EARTH)*(1/(1+e*np.cos(TA)))*np.array([np.cos(TA), np.sin(TA), 0])
    vp = constants.mu_EARTH/h*np.array([-np.sin(TA), e+np.cos(TA), 0])

    R1 = np.array([[np.cos(w), -np.sin(w), 0.],
                   [np.sin(w),  np.cos(w), 0.],
                   [0.,               0., 1.]])

    R2 = np.array([[1.,               0.,               0.],
                   [0.,      np.cos(i),     -np.sin(i)],
                   [0.,      np.sin(i),      np.cos(i)]])

    R3 = np.array([[np.cos(RA), -np.sin(RA),     0.],
                   [np.sin(RA),  np.cos(RA),     0.],
                   [0.,               0.,     1.]])

    M = R3.dot(R2.dot(R1))

    r = M.dot(rp)
    v = M.dot(vp)

    return np.concatenate((r, v))


def cart2kep(cart_elmt):
    """ Converts coordinates of a body from its cartesian coordinates into its orbitals elements (coe)
    
    Arguments : 
    
        cart_elmt (array) : States in the ECI frame

    Returns : 
        coe (array) : Body's orbital elements (SMA, ECC, INC, AOP, RAAN, TA)
    """

    R = cart_elmt[:3]
    V = cart_elmt[3:]

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(R, V) / r

    H = np.cross(R, V)
    h = np.linalg.norm(H)

    i = np.arccos(H[2] / h)

    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    if n != 0:
        RA = np.arccos(N[0]/n)
        if N[1] < 0:
            RA = 2*np.pi - RA
    else:
        RA = 0

    E = 1/constants.mu_EARTH * ((v**2 - constants.mu_EARTH/r)*R - r*vr*V)
    e = np.linalg.norm(E)

    if n != 0:
        if e > 1e-10:
            w = np.arccos(np.dot(N, E) / n / e)
            if E[2] < 0:
                w = 2*np.pi - w
        else:
            w = 0
    else:
        w = 0

    if e > 1e-10:
        TA = np.arccos(np.dot(E, R) / e / r)
        if vr < 0:
            TA = 2*np.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = np.arccos(np.dot(N, R) / n / r)
        else:
            TA = 2*np.pi - np.arccos(np.dot(N, R) / n / r)

    a = h**2 / constants.mu_EARTH / (1 - e**2)

    coe = [a, e, i, w, RA, TA]

    return coe


def kep2eqnct(coe_kep):
    """ Converts Keplerian coordinates into Equinoctial coordinates.

    Arguments : 

        coe_kep (array) : Keplerian coordinates

    Returns : 

        coe_eqnc (array) : Equinoctial coordinates

    """

    SMA, ECC, INC, AOP, RAAN, TA = coe_kep

    p = SMA * (1 - ECC**2)
    f = ECC * np.cos(RAAN + AOP)
    g = ECC * np.sin(RAAN + AOP)
    h = np.tan(INC / 2) * np.cos(RAAN)
    k = np.tan(INC / 2) * np.sin(RAAN)
    L = RAAN + AOP + TA

    return [p, f, g, h, k, L]


def eqnct2kep(coe_eqnc):
    """ Converts Equinoctial coordinates into Keplerian coordinates.

    Arguments : 

        coe_eqnc (array) : Equinoctial coordinates

    Returns : 

        coe_kep (array) : Keplerian coordinates

    """

    p, f, g, h, k, L = coe_eqnc

    SMA = p / (1 - f**2 - g**2)
    ECC = np.sqrt(f**2 + g**2)
    INC = 2 * np.arctan(np.sqrt(h**2 + k**2))
    AOP = np.arctan(g / f) - np.arctan(k / h)
    RAAN = np.arctan2(k, h)
    TA = L - (RAAN + AOP)

    return [SMA, ECC, INC, AOP, RAAN, TA]


def eqnct2kep_avg(coe_eqnc):
    """ Converts averaged Equinoctial coordinates into Keplerian coordinates.

    Arguments : 

        coe_eqnc (array) : Equinoctial coordinates

    Returns : 

        coe_kep (array) : Keplerian coordinates

    """

    p, f, g, h, k, L = coe_eqnc

    SMA = p / (1 - f**2 - g**2)
    ECC = np.sqrt(f**2 + g**2)
    INC = 2 * np.arctan(np.sqrt(h**2 + k**2))
    AOP = np.arctan(g / f) - np.arctan(k / h)
    RAAN = np.arctan2(k, h)

    return [SMA, ECC, INC, AOP, RAAN]