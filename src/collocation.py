#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 11:05:12 2020

@author: SEMBLANET Tom

"""

from scipy import interpolate
import numpy as np

import src.utils as utils


class HermiteSimpson:
    """ `HermiteSimpson` class implements the Gauss-Lobatto transcription method of order 4 allowing
        to compute the defects constraints values and to approximate the value of an integrand using
        trapezoidal quadrature.

        Parameters
        ----------
        options : dict
            Transcription and Optimization options dictionnary

        Attributs
        ---------
        options : dict
            Transcription and Optimization options dictionnary
        defects : ndarray
            Matrix of the defects constraints values

    """

    def __init__(self, options):
        """ Initialization of the `HermiteSimpson` class """
        self.options = options

        self.defects = np.ndarray(
            (self.options['n_states'], self.options['n_nodes']-1))

    def compute_states_col(self, states, controls, f_prm, f, sc_fac):
        """ Computation of the states at the collocation points

            Parameters
            ----------
            states : ndarray
                Matrix of the states at the generic points
            controls : ndarray
                Matrix of the controls at the generic points
            f_prm : array
                Array of the free parameters
            f : function
                Function of the dynamic
            sc_fac : float
                Time scaling factor

            Returns
            -------
            states_col : ndarray
                Matrix of the states at the collocation points

        """

        # Computation of the derivatives matrix
        F = sc_fac * f(states, controls, f_prm)

        # Computation of the states at collocation points
        states_col = .5 * (states[:, 1:] + states[:, :-1]) + \
            self.options['h']/8 * (F[:, :-1] - F[:, 1:])

        return states_col

    def compute_defects(self, states, controls, controls_col, f_prm, f, sc_fac):
        """ Computation of the defects constraints values using hermite-simpson method.

            Parameters
            ----------
            states : ndarray
                Matrix of states
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation nodes
            f_prm : array
                Array of the free parameters
            f : function
                Function of the dynamics
            sc_fac : float
                Scale factor

            Return
            ------
            defects : ndarray
                Matrix of the defects constraints values

        """

        # Computation of derivatives at grid-points
        F = sc_fac * f(states, controls, f_prm)

        # Computation of states at col-points
        states_col = .5 * (states[:, 1:] + states[:, :-1]) + \
            self.options['h']/8 * (F[:, :-1] - F[:, 1:])

        # Computation of derivatives at col-points
        F_col = sc_fac * f(states_col, controls_col, f_prm)

        # Computation of defects matrix
        self.defects = states[:, 1:] - states[:, :-1] - \
            self.options['h']/6 * (F[:, :-1] + 4*F_col + F[:, 1:])

        return self.defects

    def quadrature(self, f_val, f_val_col):
        """ Approximates the integrand of a funtion using hermite-simpson quadrature.

            Parameters
            ----------
            f_val : array
                Values of the function at the nodes of the time grid
            f_val_val : array
                Values of the function at the collocation nodes

            Return
            ------
            sum : float
                Integrand approximation value

        """

        sum_ = 0
        for k in range(len(f_val)-1):
            sum_ += self.options['h'][k]/6 * \
                (f_val[k] + 4*f_val_col[k] + f_val[k+1])

        return sum_

    def nodes_adaptation(self, x_i, u_i, t_i):
        """ Scales the time grid so it belongs to the interval [-1, 1]
            and computes the time-step array.

            Parameters
            ----------
            x_i : ndarray
                Matrix of the initial guess states
            u_i : ndarray
                Matrix of the initial guess controls
            t_i : array
                Array of the initial guess time grid

            Returns
            -------
            x : ndarray
                Matrix of the states
            u : ndarray
                Matrix of the controls
            h : array
                Array of the time-steps

        """

        # Time scaling
        self.scl_time = utils.scale_time(t_i)

        # Computation of time-steps
        h = self.scl_time[1:] - self.scl_time[:-1]

        # Save the time-steps array in the object's options dictionnary
        self.options['h'] = h

        # States and controls values remain the same
        x = x_i
        u = u_i

        return x, u, h
        