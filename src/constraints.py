#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 10:30:53 2020

@author: SEMBLANET Tom

"""

import numpy as np
import time

import src.utils as utils


class Constraints:
    """ `Constraints` class manages the constraints of the problem namely the path, event and
        defects constraints.
        It manages the construction of the constraints lower and upper boundaries vectors,
        the computation of the path, event and defect constraints at each iteration round.
        
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
        low : array
            Constraints vector lower boundaries
        upp : array
            Constraints vector upper boundaries
        cost : Cost
            Cost object managing the cost computation of the problem
        scaling : Scaling
            Scaling object managing the computation of the scaling factors

    """

    def __init__(self, problem, tr_method):
        """ Initialization of the `Constraints` class """

        # User-defined optimal-control problem
        self.problem = problem

        # Transcription method
        self.tr_method = tr_method

        # Computation of the equality and inequality constraints
        self.path_e_indx, self.path_i_indx, self.event_e_indx, self.event_i_indx = \
            self.equality_inequality_indx()

        # Computation of the equality and inequality constraints boundaries
        self.path_eq_bnd, self.event_eq_bnd = self.build_eq_constraints_boundaries()
        self.path_in_low_bnd, self.path_in_upp_bnd, self.event_in_low_bnd, self.event_in_upp_bnd = \
            self.build_in_constraints_boundaries()


    def equality_inequality_indx(self):
    	""" Return the index of the equality constraints as well as the index of the inequality ones """

    	path_e_indx = np.array([], dtype=np.intc)
    	path_i_indx = np.array([], dtype=np.intc)

    	event_e_indx = np.array([], dtype=np.intc)
    	event_i_indx = np.array([], dtype=np.intc)

    	for k, (path_ct_l, path_ct_u) in enumerate(zip(self.problem.low_bnd.path, self.problem.upp_bnd.path)):
    		
    		if path_ct_l == path_ct_u:
    			path_e_indx = np.append(path_e_indx, [k])
    		else:
    			path_i_indx = np.append(path_i_indx, [k])

    	for k, (event_ct_l, event_ct_u) in enumerate(zip(self.problem.low_bnd.event, self.problem.upp_bnd.event)): 

    		if event_ct_l == event_ct_u:
    			event_e_indx = np.append(event_e_indx, k)
    		else:
    			event_i_indx = np.append(event_i_indx, k)

    	return path_e_indx, path_i_indx, event_e_indx, event_i_indx


    def build_eq_constraints_boundaries(self):
        """ Construction of the vector containing the boundaries of both the path and 
            event equality constraints """

        path_eq_bnd = np.hstack([self.problem.low_bnd.path[self.path_e_indx]]
                             * (self.problem.prm['n_nodes']))

        event_eq_bnd = self.problem.low_bnd.event[self.event_e_indx]

        return path_eq_bnd, event_eq_bnd

    def build_in_constraints_boundaries(self):
        """ Construction of the vectors containing the boundaries of both the path and 
            event inequality constraints """

        path_in_low_bnd = np.hstack([self.problem.low_bnd.path[self.path_i_indx]]
                             * (self.problem.prm['n_nodes']))

        path_in_upp_bnd = np.hstack([self.problem.upp_bnd.path[self.path_i_indx]]
                             * (self.problem.prm['n_nodes']))

        event_in_low_bnd = self.problem.low_bnd.event[self.event_i_indx]
        event_in_upp_bnd = self.problem.upp_bnd.event[self.event_i_indx]

        return path_in_low_bnd, path_in_upp_bnd, event_in_low_bnd, event_in_upp_bnd

    def compute_constraints(self, decision_variables_vector):
        """ Computation of the constraints 

        Parameters
        ----------
        decision_variables_vector : array
           Decision variables vector

        Returns
        -------
        con : array
            Constraints vector.

        """

        # Unpacking the decision variables vector
        t_i, t_f, f_prm, states, controls, controls_mid = utils.unpack_decision_variable_vector(
            decision_variables_vector, self.problem.prm)

        # Update of the initial and final times as well as time scaling factor
        self.problem.prm['t_i'] = t_i
        self.problem.prm['t_f'] = t_f
        self.problem.prm['sc_factor'] = (t_f - t_i)/2

        # Computation of the defects constraints and conversio into a 1D-array
        defects = self.tr_method.compute_defects(
            states, controls, controls_mid, f_prm, self.problem.dynamics,
            self.problem.prm['sc_factor']).flatten(order='F')


        # Computation of path constraints and conversion into a 1D-array
        if self.problem.prm['n_path_con'] != 0:
            path = self.problem.path_constraints(
                states, controls, f_prm)


            if len(self.path_e_indx) != 0:
                eq_path_ct = path[self.path_e_indx, :].flatten(order='F') - self.path_eq_bnd
            else:
                eq_path_ct = np.empty(0)
            
            if len(self.path_i_indx) != 0:
                in_path_ct_low = self.path_in_low_bnd - path[self.path_i_indx, :].flatten(order='F')
                in_path_ct_upp = path[self.path_i_indx, :].flatten(order='F') - self.path_in_upp_bnd
                in_path_ct = np.concatenate((in_path_ct_low, in_path_ct_upp))
            else:
                in_path_ct = np.empty(0)

        else:
            eq_path_ct = np.empty(0)
            in_path_ct = np.empty(0)

        # Computation of the event constraints 
        if self.problem.prm['n_event_con'] != 0:
            event = self.problem.event_constraints(states[:, 0], controls[:, 0], states[:, -1],
                                                   controls[:, -1], f_prm, t_i, t_f)

            if len(self.event_e_indx) != 0:
                eq_event_ct = event[self.event_e_indx] - self.event_eq_bnd
            else:
                eq_event_ct = np.empty(0)
            
            if len(self.event_i_indx) != 0:
                in_event_ct_low = self.event_in_low_bdn - path[self.event_i_indx]
                in_event_ct_upp = path[self.event_i_indx] - self.event_in_upp_bnd  
                in_event_ct = np.concatenate((in_event_ct_low, in_event_ct_upp))
            else:
                in_event_ct = np.empty(0)

        else:
            event = np.empty(0)

        return np.concatenate((defects, eq_path_ct, eq_event_ct, in_path_ct, in_event_ct))
