from mpi4py import MPI
import cppad_py
import numpy as np
import matplotlib.pyplot as plt

from problem import Problem
from optimization import Optimization

from mr_scl.dynamics import equinoctial_averaged_derivatives
from mr_scl.constants import R_EARTH, mu_EARTH, g0, mu_VESTA, R_VESTA
from mr_scl.coc import kep2eqnct


class MR_Minimum_Time(Problem):
    """ Multi-Revolution | Minimum time : Optimal Control Problem """

    def __init__(self, comm, rank, size):
        """ Initialization of the `MR_Minimum_Time` class """
        n_states = 6
        n_controls = 5
        n_path_con = 1
        n_event_con = 8
        n_f_par = 0
        n_nodes = 64

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

        # Multiprocessing related variables
        self.comm = comm
        self.rank = rank 
        self.size = size

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
        ECC  = 0.01

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
            (self.prm['n_path_con'], self.prm['n_nodes']), dtype=cppad_py.a_double)

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
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

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



    def dynamics(self, states, controls, f_prm, mapping_func=False):
        """ Computation of the states derivatives """
        dynamics = np.ndarray(
            (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)

        if mapping_func == True:
            for i in range(dynamics.shape[1]):
                dynamics[:, i] = equinoctial_averaged_derivatives(t=0, \
                    y_bar=states[:, i], costates=controls[:, i], Isp=self.Isp, T_max=self.T_max, delta=self.delta, \
                    mu_scl=self.mu_scl, g0_scl=self.g0_scl, N=50)

        else:
            # Send the states, controls and free-parameters to each process
            for k in range(self.size)[1:]:
                indx_b = int( k    * (states.shape[1] / self.size))
                indx_e = int((k+1) * (states.shape[1] / self.size))

                self.comm.send(obj={'states': states[:, indx_b:indx_e], \
                                    'controls': controls[:, indx_b:indx_e], \
                                    'f_prm': f_prm}, dest=k)

            # Compute a part of the dynamics 
            indx_e = int(states.shape[1] / self.size)
            dynamics[:, :indx_b] = self.dynamics_worker(states=states[:, :indx_e], \
                                                          controls=controls[:, :indx_e], \
                                                         f_prm=f_prm)

            # Gather the results from each process
            for k in range(self.size)[1:]:
                indx_b = int( k    * (states.shape[1] / self.size))
                indx_e = int((k+1) * (states.shape[1] / self.size))

                dynamics[:, indx_b:indx_e] = self.comm.recv(source=k)

        return dynamics

    def dynamics_worker(self, states, controls, f_prm):
        """ Computation of the states derivatives by a worker (multiprocessing implementation) """
        dynamics = np.ndarray(
            (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)

        for i in range(dynamics.shape[1]):
            dynamics[:, i] = equinoctial_averaged_derivatives(t=0, \
                y_bar=states[:, i], costates=controls[:, i], Isp=self.Isp, T_max=self.T_max, delta=self.delta, \
                mu_scl=self.mu_scl, g0_scl=self.g0_scl, N=50)

        return dynamics

    def wait_and_cpt(self):
        """ Function dedicated to workers processing """

        while True:
            message = self.comm.recv(source=0)

            if message == 'end':
                break

            else:
                dynamics = self.dynamics_worker(states=message['states'], \
                                             controls=message['controls'], \
                                             f_prm=message['f_prm'])
                self.comm.send(obj=dynamics, dest=0)


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


if __name__ == '__main__':

    # Initialization of the communicators for multiprocessing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Instantiation of the problem
    problem = MR_Minimum_Time(comm=comm, rank=rank, size=size)

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # The main process runs the optimization
    if rank == 0:
        optimization.run()

        # Once optimization is finished, free the workers processes
        for k in range(size)[1:]:
            comm.send(obj='end', dest=k)

    else:
        problem.wait_and_cpt()
