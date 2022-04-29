import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.problem import Problem

from mr_scl.dynamics import equinoctial_averaged_derivatives
from mr_scl.constants import R_EARTH, mu_EARTH, g0, mu_VESTA, R_VESTA
from mr_scl.coc import kep2eqnct


class MR_Minimum_Time(Problem):
	""" Multi-Revolution | Minimum time : Optimal Control Problem """

	def __init__(self):
		""" Initialization of the `MR_Minimum_Time` class """
		n_states = 6
		n_controls = 5
		n_path_con = 1
		n_event_con = 8
		n_f_par = 0
		n_nodes = 64

		Problem.__init__(self, n_states, n_controls, n_path_con,
						 n_event_con, n_f_par, n_nodes)

	def set_constants(self):
		""" Setting of the problem constants """

		# Characteristic parameters
		# =========================

		# Characteristic time, length and mass ([s], [km], [kg])
		self.t_ref = 86400.000
		self.l_ref = 42241.096
		self.m_ref =  2000.000

		# Characteristic velocity and acceleration ([km/s], [km/s^2])
		self.v_ref = self.l_ref / self.t_ref
		self.a_ref = self.v_ref / self.t_ref

		# Scaled constants
		# ================

		self.mu_scl = mu_EARTH * self.t_ref**2 / self.l_ref**3
		self.g0_scl = g0 * self.t_ref**2 / self.l_ref


		# # Characteristic parameters
		# # =========================

		# # Characteristic time, length and mass ([s], [km], [kg])
		# self.t_ref = 47094.465
		# self.l_ref =  1000.000
		# self.m_ref =  2000.000

		# # Characteristic velocity and acceleration ([km/s], [km/s^2])
		# self.v_ref = self.l_ref / self.t_ref
		# self.a_ref = self.v_ref / self.t_ref

		# # Scaled constants
		# # ================

		# self.mu_scl = mu_VESTA * self.t_ref**2 / self.l_ref**3
		# self.g0_scl = g0 * self.t_ref**2 / self.l_ref



		# Initial orbit (Keplerian parameters)
		# ======================================

		SMA  = ( 7000 ) / self.l_ref
		ECC  = 0.01
		INC  = ( 0.05 ) * np.pi / 180
		AOP  = ( 0.00 ) * np.pi / 180
		RAAN = ( 0.00 ) * np.pi / 180
		TA   = 0

		# Computation of the equinoctial elements
		self.pi, self.fi, self.gi, self.hi, self.ki, _ = kep2eqnct(np.array([SMA, ECC, INC, AOP, RAAN, TA]))

		# Final orbit (Equinoctial parameters)
		# ====================================

		SMA  = ( 42000 ) / self.l_ref
		ECC  = 0.001

		self.a_f = SMA
		self.e_f = ECC

		# Computation of the equinoctial elements
		self.pf, self.ff, self.gf, self.hf, self.kf, _ = kep2eqnct(np.array([SMA, ECC, INC, AOP, RAAN, TA]))


		# Spacecraft
		# ==========

		self.mi = 300 / self.m_ref
		self.m_dry = 10 / self.m_ref
		self.T_max = 1e-3 / self.m_ref / self.a_ref
		self.Isp = 3100 / self.t_ref
		self.delta = 1.

		# Time
		# ====
		self.tof = ( 15 * 86400 ) / self.t_ref
		self.tf_max = ( 365 * 86400 ) / self.t_ref


	def set_boundaries(self):
		""" Setting of the states, controls, free-parameters, initial and final times
						boundaries """

		# `p` 
		self.low_bnd.states[0] = 0
		self.upp_bnd.states[0] = +3E5 / self.l_ref

		# `f`
		self.low_bnd.states[1] = -1.01
		self.upp_bnd.states[1] = +1.01

		# `g`
		self.low_bnd.states[2] = -1.01
		self.upp_bnd.states[2] = +1.01

		# `h`
		self.low_bnd.states[3] = -1.01
		self.upp_bnd.states[3] = +1.01

		# `k`
		self.low_bnd.states[4] = -1.01
		self.upp_bnd.states[4] = +1.01

		# Mass
		self.low_bnd.states[5] = self.m_dry
		self.upp_bnd.states[5] = self.mi

		# Definition of control variables boundaries
		# ------------------------------------------

		# lambda_p 
		self.low_bnd.controls[0] = -1.01
		self.upp_bnd.controls[0] = +1.01

		# lambda_f 
		self.low_bnd.controls[1] = -1.01
		self.upp_bnd.controls[1] = +1.01

		# lambda_g 
		self.low_bnd.controls[2] = -1.01
		self.upp_bnd.controls[2] = +1.01

		# lambda_h 
		self.low_bnd.controls[3] = -1.01
		self.upp_bnd.controls[3] = +1.01

		# lambda_k
		self.low_bnd.controls[4] = -1.01
		self.upp_bnd.controls[4] = +1.01

		# Definition of initial / final times
		# -----------------------------------

		# Initial time [TO REMOVE]
		self.low_bnd.ti = self.upp_bnd.ti = 0

		# Final time 
		self.low_bnd.tf = 60 / self.t_ref
		self.upp_bnd.tf = self.tf_max


	def path_constraints(self, states, controls, f_par):
		""" Computation of the path constraints """
		constraints = np.ndarray(
			(self.prm['n_path_con'], self.prm['n_nodes']))

		lambda_p, lambda_f, lambda_g, lambda_h, lambda_k = controls

		constraints[0] = lambda_p*lambda_p + lambda_f*lambda_f + lambda_g*lambda_g + \
			lambda_h*lambda_h + lambda_k*lambda_k

		return constraints

	def set_path_constraints_boundaries(self):
		""" Setting of the path constraints boundaries """

		# Uniqueness of the costates
		self.low_bnd.path[0] = self.upp_bnd.path[0] = 1


	def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
		""" Computation of the events constraints """
		events = np.ones(shape=self.prm['n_event_con'])

		# Initial conditions
		pi, fi, gi, hi, ki, mi = xi
		pf, ff, gf, hf, kf, _  = xf

		ff2 = ff*ff
		gf2 = gf*gf
		hf2 = hf*hf 
		kf2 = kf*kf

		# Final condition
		SMA_CT    = pf / (1 - ff2 - gf2)
		ECC_CT    = ff2 + gf2

		events[0]  = pi  
		events[1]  = fi  
		events[2]  = gi  
		events[3]  = hi  
		events[4]  = ki  
		events[5]  = mi

		events[6]    = SMA_CT 
		events[7]    = ECC_CT

		return events

	def set_events_constraints_boundaries(self):
		""" Setting of the events constraints boundaries """

		# Initial `p`
		self.low_bnd.event[0] = self.upp_bnd.event[0] = self.pi

		# Initial `f`
		self.low_bnd.event[1] = self.upp_bnd.event[1] = self.fi

		# Initial `g`
		self.low_bnd.event[2] = self.upp_bnd.event[2] = self.gi

		# Initial `h`
		self.low_bnd.event[3] = self.upp_bnd.event[3] = self.hi

		# Initial `k`
		self.low_bnd.event[4] = self.upp_bnd.event[4] = self.ki

		# Initial `m`
		self.low_bnd.event[5] = self.upp_bnd.event[5] = self.mi

		# Final SMA
		self.low_bnd.event[6] = self.upp_bnd.event[6] = self.a_f

		# Final ECC
		self.low_bnd.event[7] = self.upp_bnd.event[7] = self.e_f**2


	def dynamics(self, states, controls, f_prm, expl_int=False):
		""" Computation of the states derivatives """
		dynamics = np.ndarray(
			(states.shape[0], states.shape[1]))



		# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

		for i in range(dynamics.shape[1]):
			dynamics[:, i] = equinoctial_averaged_derivatives(t=0, \
				y_bar=states[:, i], costates=controls[:, i], Isp=self.Isp, T_max=self.T_max, delta=self.delta, \
				mu_scl=self.mu_scl, g0_scl=self.g0_scl)


		# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~ ~ ~ ~ ~ ~ ~ 

		return dynamics

	def end_point_cost(self, ti, xi, tf, xf, f_prm):
		""" Computation of the end point cost (Mayer term) """

		# Minimization of the transfer time
		return tf 

	def set_initial_guess(self):
		""" Setting of the initial guess for the states, controls, free-parameters
						and time grid """

		# Time
		# ====
		self.initial_guess.time = np.linspace(
			0, self.tof, self.prm['n_nodes'])

		# States
		# ======
		self.initial_guess.states = np.ndarray(
			shape=(self.prm['n_states'], self.prm['n_nodes']))

		# Parameter `p`
		self.initial_guess.states[0] = np.linspace(
			self.pi, self.pf, self.prm['n_nodes'])

		# Parameter `f`
		self.initial_guess.states[1] = np.linspace(
			self.fi, self.ff, self.prm['n_nodes'])

		# Parameter `g`
		self.initial_guess.states[2] = np.linspace(
			self.gi, self.gf, self.prm['n_nodes'])

		# Parameter `h`
		self.initial_guess.states[3] = np.linspace(
			self.hi, self.hf, self.prm['n_nodes'])

		# Parameter `k`
		self.initial_guess.states[4] = np.linspace(
			self.ki, self.kf, self.prm['n_nodes'])

		# Parameter `m`
		self.initial_guess.states[5] = np.linspace(
			self.mi, max(self.m_dry, self.mi - self.T_max / self.Isp / self.g0_scl * self.tof), self.prm['n_nodes'])

		# Controls
		# ========
		self.initial_guess.controls = np.ndarray(
			shape=(self.prm['n_controls'], self.prm['n_nodes']))

		# Costate `lambda_p`
		self.initial_guess.controls[0] = np.linspace(
			+1.0, +1.0, self.prm['n_nodes'])

		# Costate `lambda_f`
		self.initial_guess.controls[1] = np.linspace(
			+0.0, +0.0, self.prm['n_nodes'])

		# Costate `lambda_g`
		self.initial_guess.controls[2] = np.linspace(
			+0.0, +0.0, self.prm['n_nodes'])

		# Costate `lambda_h`
		self.initial_guess.controls[3] = np.linspace(
			+0.0, +0.0, self.prm['n_nodes'])

		# Costate `lambda_k`
		self.initial_guess.controls[4] = np.linspace(
			+0.0, +0.0, self.prm['n_nodes'])



# if __name__ == '__main__':

# 	# Instantiation of the problem
# 	problem = MR_Minimum_Time()

# 	# Instantiation of the optimization
# 	optimization = Optimization(problem=problem)

# 	# Launch of the optimization
# 	optimization.run()
