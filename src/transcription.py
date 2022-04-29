#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 09:50:23 2020

@author: SEMBLANET Tom

"""

import numpy as np

import src.constraints as cs, src.cost as ct, src.collocation as col, src.scaling as sc, src.utils as utils


class Transcription:
    """ `Transcription` class translates a user-defined optimal control problem into a parametric optimization
        which can be solved by means of Non-Linear Programming Solvers (IPOPT or SNOPT). 

        Parameters
        ----------
        problem : Problem
            Optimal-control problem defined by the user

        Attributes
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        decision_variables_vector : array
            Vector of the problem variables (states, controls, free-parameters, time, path constraints and event constraints)
        decision_variables_vector_low : array
            Decision variables vector lower boundaries 
        decision_variables_vector_upp : array
            Decision variables vector upper boundaries 
        constraints : Constraints
            Constraints object managing the constraints computation of the problem
        cost : Cost
            Cost object managing the cost computation of the problem
        scaling : Scaling
            Scaling object managing the computation of the scaling factors
        tr_method : Collocation or Pseudospectral
            Transcription method object used to compute the defects constraints

    """

    def __init__(self, problem):
        """ Initialization of the `Transcription` class """

        # User-defined optimal-control problem
        self.problem = problem

        # States, Controls and Time transformation
        states, controls = self.nodes_adaptation()

        # Decision variable vector construction
        self.decision_variables_vector = self.build_decision_variable_vector(states,
                                                                             controls)

        # Decision variables vector boundaries construction
        self.decision_variables_vector_low, self.decision_variables_vector_upp = \
            self.build_decision_variable_vector_boundaries()

        # Constraints object instanciation + constraints
        # boundaries and Jacobian construction
        self.constraints = cs.Constraints(
            problem=self.problem, tr_method=self.tr_method)

        # Cost object instantiation + cost Gradient construction
        self.cost = ct.Cost(problem=self.problem, tr_method=self.tr_method)

        # # Scaling factors computation
        # cost_gradient = self.cost.compute_cost_gradient(
        #     self.decision_variables_vector)
        # self.constraints.compute_constraints_jacobian(
        #     self.decision_variables_vector)
        # self.scaling = sc.Scaling(self.decision_variables_vector_low, self.decision_variables_vector_upp,
        #                           self.constraints.jac_dict['jac_data'], cost_gradient, self.problem.prm)

    def nodes_adaptation(self):
        """ Scales the time so it belongs to the interval [-1, 1]
            If pseudospectral method is used computation of the LGL and CGL nodes and
            states and controls are interpolated 

            Returns
            -------
            states : ndarray
                Matrix of the states variables
            controls : ndarray
                Matrix of the controls variables 

        """

        x_i = self.problem.initial_guess.states
        u_i = self.problem.initial_guess.controls
        t_i = self.problem.initial_guess.time

        # Instanciation and assignment of transcription method to the phase
        self.tr_method = col.HermiteSimpson(self.problem.prm.copy())

        # Computation of new States, Controls, Times according to the
        # choosen transcription method
        states, controls, h = self.tr_method.nodes_adaptation(
            x_i, u_i, t_i)

        # Stocks the information of unscaled t_i, t_f,
        # scaled time-steps and scale factor
        self.problem.prm['h'] = h
        self.problem.prm['t_i'] = t_i[0]
        self.problem.prm['t_f'] = t_i[-1]

        self.problem.prm['sc_factor'] = (t_i[-1] - t_i[0])/2

        return states, controls

    def build_decision_variable_vector(self, states_mat, controls_mat):
        """ Construction of the decision variables vector 

        Parameters
        ----------
        states_mat : ndarray
            Matrix of the states variables
        controls_mat : ndarray
            Matrix of the controls variables

        Returns
        -------
        dvv : array
            Decision variables vector containing the states, controls, free parameters, initial and final time

        """

        # Computation of the phase decision variables vector
        dvv = utils.make_decision_variable_vector(states_mat, controls_mat, self.problem.initial_guess.controls_col,
                                                  self.problem.initial_guess.time[0], self.problem.initial_guess.time[-1],
                                                  self.problem.initial_guess.f_prm, self.problem.prm)

        return dvv

    def build_decision_variable_vector_boundaries(self):
        """ Construction of the decision variables vector lower and upper boundaries 

        Returns
        -------
        low : array
            Decision variables vector lower boundaries
        upp : array
            Decision variables vector upper boundaries 

        """

        # States boundaries initialization
        states_low = np.hstack(
            [self.problem.low_bnd.states] * self.problem.prm['n_nodes'])
        states_upp = np.hstack(
            [self.problem.upp_bnd.states] * self.problem.prm['n_nodes'])

        # Controls boundaries initialization
        controls_low = np.hstack(
            [self.problem.low_bnd.controls] * self.problem.prm['n_nodes'])
        controls_upp = np.hstack(
            [self.problem.upp_bnd.controls] * self.problem.prm['n_nodes'])

        # Controls-mid boundaries initizalization
        controls_col_low = np.hstack(
            [self.problem.low_bnd.controls] * (self.problem.prm['n_nodes']-1))
        controls_col_upp = np.hstack(
            [self.problem.upp_bnd.controls] * (self.problem.prm['n_nodes']-1)) 

        # Concatenation of the states, controls, controls-mid, free-parameters, initial and final time boundaries
        low = np.concatenate((states_low, controls_low, controls_col_low, self.problem.low_bnd.f_par,
                              [self.problem.low_bnd.ti], [self.problem.low_bnd.tf]))
        upp = np.concatenate((states_upp, controls_upp, controls_col_upp, self.problem.upp_bnd.f_par,
                              [self.problem.upp_bnd.ti], [self.problem.upp_bnd.tf]))

        return low, upp
