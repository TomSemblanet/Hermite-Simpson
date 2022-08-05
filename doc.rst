"""""""""""""""""""""""
COLIBRI's documentation
"""""""""""""""""""""""

==================================
Optimal control problem definition
==================================

As a reminder, in its generic form an optimal control problem is defined as follows:

.. math::

   \\frac{ \\sum_{t=0}^{N}f(t,k) }{N}

:math:`\\frac{1}{2}`

With N the number of phases, minimize the cost function:

:math:`J = \sum_{i=1}^N \Phi_i(x(t_0), t_0, x(t_f), t_f; p) + \int_{t_0}^{t_f} \mathcal{L}_i(x(t), u(t), t; p) \mathrm{d}t `

where :math:`\Phi` is named the Mayer term and :math:`\mathcal{L}` the Legendre term.
Subject to the constraints : 

:math:`\dot{x}_i(t) = f_i(x_i(t), u_i(t), t; p_i) `
:math:`\bold{h}_i(x_i(t), u_i(t), t; p_i) \leq \bold{C_h}_i`                     
:math:`\bold{\phi}_i(x_i(t_0), t_0, x_i(t_f), t_f; p_i) \leq \bold{C_{\phi_i}}`
:math:`\bold{\L}_i(x_i(t_0), x_i(t_f), p_i, x_{i+1}(t_0), x_{i+1}(t_f), p_{i+1}) \leq \bold{C_{L_i}}`

respectively the
- dynamics constraints : imposing that the states respect the equations of motion of the problem (example: Keplerian equation of motion)
- path constraints : imposing constraints on the entire trajectory (example: Distance between a Spacecraft and the Earth :math:`\geq 10,000` km )
- boundaries constraints : imposing constraints on the initial and / or final states, controls and time (example : Fixe the initial state of a Spacecraft)
- linkage constraints : imposing constraints between the phase (example : The state of a Spacecraft at the end of the phase n°1 must be equal to the state 
of the Spacecraft at the beginning of the phase n°2)


COLIBRI makes it possible to define all the components of a phase previously named : 
- Mayer and Legendre terms of the cost function
- Dynamics constraints (EOMs)
- Path constraints (equality and/or inequality)
- Boundary constraints (equality and/or inequality)
- Linkage constraints (equality and/or inequality)

using differents methods that the user must fill.

==============================
Optimal control representation
==============================
 
Following the generic definition of an optimal control problem, the implementation of a problem in COLIBRI is done by defining separatly each phases.

A phase is represented by a Python class inheriting from the ``Phase`` class and the different features of the phase (cost function, dynamical, path, 
boundary constraints etc ...) are defined through Python methods (see Phase definition chapter for implementation).

The phases thus created are then gathered as attributes of a superclass inheriting from the ``Problem`` class. This superclass allows to define the 
linkage constraints between the phases (see Problem definition chapter for implementation).

.. image:: images/problem_implementation.png


================
Phase definition
================

This section describes the definition and implementation of a Phase from the construction of the class to the definition of the 
differents methods (several examples of problems can be found here). In the following, several Python code are provided illustrating the 
implementation of the class and its different methods. 
The user doesn't have to deal neither with the parameters passed to the methods nor to the several additional lines of code found in the 
methods definition. To implement his optimal control problem, the user only has to fill the part of the code represented with the ``...`` and 
he also has to potentially duplicate some lines of code.


--------------------
Phase initialization
--------------------

As previously stated, each Phase is represented by a Python class inheriting from the ``Phase`` class. 

When building the class, several parameters must be passed : 
    - Number of states 
    - Number of controls
    - Number of free parameters
    - Number of nodes
    - Number of equality boundary constraints
    - Number of inequality boundary constraints 
    - Number of equality path constraints 
    - Number of inequality path constraints

.. code:: python

   class MyPhase(Phase):
    """ Example of phase instantiation """

    def __init__(self):
        """ Initialization of `MyPhase` class """
        N_states       = ...
        N_controls     = ... 
        N_param        = ...
        N_nodes        = ... 
        N_eq_event_con = ...
        N_in_event_con = ...
        N_eq_path_con  = ...
        N_in_path_con  = ...

        Phase.__init__(self, N_states, N_controls, N_param, N_nodes, N_eq_path_con,
                         N_eq_event_con, N_in_path_con, N_in_event_con)


Each phase class includes several methods defining :
    - the states, controls, parameters, initial and final time boundaries [mandatory]
    - the cost function (Mayer and/or Legendre terms) [optional]
    - the dynamics constraints (EOMs of the phase) [mandatory]
    - the path equality and/or inequality constraints [optional]
    - the boundaries equality and/or inequality constraints [optional]
    - the initial guess for the states, controls, parameters and time grid [mandatory]

In the following, we describe in detail the implementation of each of these methods.

-----------------
Boundaries values
-----------------

The states, controls, parameters, initial and final time boundaries are defined through the ``set_boundaries`` method : 

.. code:: python

    def set_boundaries(self):
        """ this methods allows to set the boundary values of the states, controls, free parameters
            initial and final time """

        # States boundaries
        self.dv_low_bnd.states[...] = ...
        self.dv_upp_bnd.states[...] = ...

        # Controls boundaries
        self.dv_low_bnd.controls[...] = ...
        self.dv_upp_bnd.controls[...] = ...

        # Free parameters boundaries
        self.dv_low_bnd.param[...] = ...
        self.dv_upp_bnd.param[...] = ...

        # Initial and final times boundaries
        self.dv_low_bnd.ti = ...
        self.dv_upp_bnd.ti = ...

        self.dv_low_bnd.tf = ...
        self.dv_upp_bnd.tf = ...

where the index between brackets is the index of the corresponding state, control or parameter. 

-------------
Cost function
-------------

As the cost function can be composed by either the Mayer term or the Legendre term or both or none of them, two functions are 
implementable and both are optional : 

.. code:: python

    def Mayer_cost(self, withPartials, xi, ui, xf, uf, param, ti, tf):
        """ Computation of the end-point cost (Mayer term) """
        mayer_cost = 0
        if withPartials:
            mayer_cost = ad.Scalar(mayer_cost, True)

        mayer_cost = ...
        
        return mayer_cost

.. code:: python

    def Legendre_cost(self, withPartials, states, controls, param):
        """ implementation of the Legendre term of the cost function """
        N_nodes = br.shape(states)[1]
        legendre_cost = np.empty(shape=N_nodes)
        if withPartials:
            legendre_cost = ad.Vector(legendre_cost, True)

        for k in range(N_nodes):
            legendre_cost[k] = ...

        return legendre_cost

where each value of the ``legendre_cost`` vector is the expression of the `\mathcal{L}(x(t), u(t), t; p)` value in the integral.


Some simple examples are given : 

- Minimization of the product of the :math:`1^{st}` final state and the :math:`2^{nd}` final control (using the Mayer cost method):
:math:`\Phi(x(t_0), t_0, x(t_f), t_f; p) = x_1 * u_2` :
.. code:: python

    mayer_cost = xf[0] * uf[1]

- Minimization of the final time value (using the Mayer cost method): :math:`\Phi(x(t_0), t_0, x(t_f), t_f; p) = t_f` :

.. code:: python
    mayer_cost = tf

- Minimization of the total distance travelled (using the Legendre cost method): :math:`\int_{t_0}^{t_f} \dot{x}(t)^2 + \dot{y}(t)^2 \mathrm{d}t` :

.. code:: python
    for k in range(N_nodes):
            legendre_cost[k] = states[0]**2 + states[1]**2


---------------------
Dynamical constraints
---------------------

The dynamical constraints implements the Equations of Motion (EOMs) of the problem and are defined through the ``dynamics`` function :

.. code:: python

    def dynamics(self, states, controls, param, withPartials):
        """ Computation of the states derivatives """
        dynamics = np.empty(shape=len(states))
        if withPartials:
            dynamics = ad.Vector(dynamics, True)

        dynamics[...] = ...

        return dynamics

where each component of the ``dynamics`` vector is the expression of the :math: `i^{th}` state derivative.

----------------
Path constraints
----------------

Path constraints can either be inequality or equality constraints : 
    - Equality constraint : defined under the form :math:`h(x(t), u(t), t; p) - \gamma = 0`
    - Inequality constraint : defined under the form :math:`h(x(t), u(t), t; p) - \gamma \leq 0`

where :math:`\gamma` is the RAS value.

They are defined through two different functions and are both optional 

.. code:: python

    def equality_path_constraints(self, states, controls, param, withPartials):
        """ Computation of the equality path constraints """
        eq_path_con = np.empty(shape=self.charac['N_eq_path_con'])
        if withPartials:
            eq_path_con = ad.Vector(eq_path_con, True)

        eq_path_con[...] = ...

        return eq_path_con

.. code:: python

    def inequality_path_constraints(self, states, controls, param, withPartials):
        """ Computation of the inequality path constraints """
        in_path_con = np.empty(shape=self.charac['N_in_path_con'])
        if withPartials:
            in_path_con = ad.Vector(in_path_con, True)

        in_path_con[...] = ...

        return in_path_con


--------------------
Boundary constraints
--------------------

Boundary constraints can either be inequality or equality constraints : 
    - Equality constraint : defined under the form :math:`\phi(x(t_0), t_0, x(t_f), t_f; p) - \gamma = 0`
    - Inequality constraint : defined under the form :math:`\phi(x(t_0), t_0, x(t_f), t_f; p) - \gamma \leq 0`

where :math:`\gamma` is the RAS value.

They are defined through two different functions and are both optional 

.. code:: python

    def equality_event_constraints(self, xi, ui, xf, uf, param, ti, tf, withPartials):
        """ Computation of the equality events constraints """
        eq_events_con = np.empty(shape=self.charac['N_eq_event_con'])
        if withPartials:
            eq_events_con = ad.Vector(eq_events_con, True)

        eq_events_con[...] = ...

        return eq_events_con

.. code:: python 

    def inequality_event_constraints(self, xi, ui, xf, uf, param, ti, tf, withPartials):
        """ Computation of the inequality events constraints """
        in_events_con = np.empty(shape=self.charac['N_in_event_con'])
        if withPartials:
            in_events_con = ad.Vector(in_events_con, True)

        in_events_con[...] = ...

        return in_events_con
