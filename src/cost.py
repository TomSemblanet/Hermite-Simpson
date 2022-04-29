#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 11:05:12 2020

@author: SEMBLANET Tom

"""

import numpy as np

import src.utils as utils


class Cost:
    """ `Cost` class manages the costs of the problem namely the Mayer and Legendre parts
        and computes the cost gradient.

        Parameters
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        tr_method : Collocation or Pseudospectral
            Transcription method object used to compute the defects constraints

        Attributes
        ----------
        problem : Problem
            Optimal-control problem defined by the user

    """

    def __init__(self, problem, tr_method):
        """ Initialization of the `Cost` class """

        # User-defined optimal-control problem
        self.problem = problem

        # Transcription method
        self.tr_method = tr_method

    def compute_cost(self, decision_variables_vector):
        """ Computation of the cost as the sum of both the Mayer and Legendre terms.


        Parameters
        ----------
        decision_variables_vector : array
            Vector of decision variables

        Returns
        -------
        cost_sum : float
            Cost value

        """

        # Initialization of the cost value
        cost_sum = 0

        # Extraction of the states & controls matrices, free parameters, initial et final times
        t_i, t_f, f_prm, states, controls, controls_mid = utils.unpack_decision_variable_vector(
            decision_variables_vector, self.problem.prm)

        # Update of the initial and final times as well as time scaling factor
        self.problem.prm['t_i'] = t_i
        self.problem.prm['t_f'] = t_f
        self.problem.prm['sc_factor'] = (t_f - t_i)/2

        # Computation of the end point cost value (ie. Mayer term)
        if hasattr(self.problem, 'end_point_cost'):
            x_i = states[:, 0]
            x_f = states[:, -1]
            cost_sum += self.problem.end_point_cost(t_i, x_i, t_f, x_f, f_prm)

        # Computation of the integrand cost value (ie. Legendre term)
        if hasattr(self.problem, 'integrand_cost'):
            f_val = self.problem.integrand_cost(states, controls, f_prm)

            # Computation of states at mid-points
            states_mid = self.tr_method.compute_states_col(
                states, controls, f_prm, self.problem.dynamics, self.problem.prm['sc_factor'])

            # Computation of integrand cost at mid-points
            f_val_mid = self.problem.integrand_cost(
                states_mid, controls_mid, f_prm)

            cost_sum += self.problem.prm['sc_factor'] * \
                self.tr_method.quadrature(
                    f_val, f_val_mid)

        return cost_sum
