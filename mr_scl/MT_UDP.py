import matplotlib.pyplot as plt

import pygmo as pg 
import numpy as np

from mr_scl.dynamics import equinoctial_averaged_derivatives
from mr_scl.constants import R_EARTH, mu_EARTH, g0, mu_VESTA, R_VESTA
from mr_scl.coc import kep2eqnct

from mr_scl.minimum_time import MR_Minimum_Time

from src.problem import Problem
from src.transcription import Transcription
from src.utils import make_decision_variable_vector

class MT_UDP:
    """ Implementation of the Multi-Revolution transfer problem using the `PyGMO` library and 
        the collocation method for transcription """

    def __init__(self, problem_tr):
        """ Initialization of the User-Defined Problem (UDP) """

        # Instantiation of the multi-revolution problem and its transcription features
        self.problem_tr = problem_tr


    def fitness(self, x):
        """ Returns the fitness of the problem """

        # Objective function
        cost = self.problem_tr.cost.compute_cost(x)

        # Constraints (defects + path + events)
        constraints = self.problem_tr.constraints.compute_constraints(x)

        return np.concatenate(([cost], constraints))

    
    def get_bounds(self):
        """ Returns the bounds of the decision variables vector (`lb` and `ub` vectors) """

        return (self.problem_tr.decision_variables_vector_low, self.problem_tr.decision_variables_vector_upp)

    def get_nobj(self):
      """ Returns the number of objectives """
      return 1

    def get_nec(self):    
      """ Returns the number of equality constraints """ 
      return len(self.problem_tr.constraints.path_e_indx) * self.problem_tr.problem.prm['n_nodes'] + \
               len(self.problem_tr.constraints.event_e_indx)

    def get_nic(self):
        """ Returns the number of inequality constraints """
        return 2 * (len(self.problem_tr.constraints.path_i_indx) * self.problem_tr.problem.prm['n_nodes'] + \
            len(self.problem_tr.constraints.event_i_indx))


def set_initial_vector(initial_guess, pb_prm):
    """ Define the initial vector by flattening and concatenating the initial guess
        matrices """
    return make_decision_variable_vector(states=initial_guess.states, controls=initial_guess.controls, \
        controls_col=initial_guess.controls_col, ti=initial_guess.time[0], tf=initial_guess.time[-1], \
        prm=initial_guess.f_prm, pb_prm=pb_prm)


if __name__ == '__main__':

    # Instantiation of the problem
    mt = MR_Minimum_Time()
    mt.setup()

    # Creation of the initial vector
    x0 = set_initial_vector(initial_guess=mt.initial_guess, pb_prm=mt.prm)

    # Creation of the Problem with all transcription features
    mt_transcribed = Transcription(problem=mt)

    # Creation of the UDP
    mt_UDP = MT_UDP(problem_tr=mt_transcribed)

    # Initialization of the population
    population = pg.population(prob=mt_UDP, size=1)
    population.set_x(0, x0)

    # Algorithm
    algorithm = pg.ipopt() 

    population = algorithm.evolve(population)
