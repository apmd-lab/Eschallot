import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
import optimization.CP_PSO as opt

var_range = np.array([[50000,500000], # film thickness
                      [0.1,0.7], # total particle volume fraction
                      [0.0,1.0], # relative number density of particle type 1
                      [50,500], # mean radius of particle type 1
                      [50,500], # mean radius of particle type 2
                      [0.1,10], # scale parameter of particle type 1
                      [0.1,10]]) # scale parameter of particle type 2

opt.CP_PSO(mode="max", swarm_size=10, var_range=var_range, N_disc=2, N_choice=4,
           iteration_limit=100, stop_limit=10, c1=[1.5,0.5], c2=[1.5,0.5], w=[0.9,0.9],
           starting_iteration=1, stop_count=0)
