import numpy as np 
import pickle

from mr_scl.coc import eqnct2kep_avg


# Path of the file containing Pickled objects
fl_path = ""
file = open(fl_path, 'r')

# Loading the content of the Pickle file
results = file.read()

states = results['opt_st']

keplerian_states = np.ndarray(shape=states.shape)

for i in range(states.shape[1]):
    keplerian_states[:, i] = eqnct2kep_avg(states[:, i])

