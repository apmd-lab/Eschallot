import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-11])

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

def gamma_distribution(N_bins, f_num, shape, scale):
    N_layer, N_type = shape.shape
    r_min_list = np.zeros((N_layer, N_type))
    r_max_list = np.zeros((N_layer, N_type))
    for n_l in range(N_layer):
        for n_t in range(N_type):
            r_min_list[n_l,n_t] = gamma.ppf(0.01, shape[n_l,n_t], loc=0, scale=1/scale[n_l,n_t])
            r_max_list[n_l,n_t] = gamma.ppf(0.99, shape[n_l,n_t], loc=0, scale=1/scale[n_l,n_t])
        
    r = np.linspace(np.min(r_min_list), np.max(r_max_list), N_bins)
    
    pdf = np.zeros((N_layer, N_type, N_bins))
    for n_l in range(N_layer):
        for n_t in range(N_type):
            pdf[n_l,n_t,:] = f_num[n_l,n_t]*gamma.pdf(r, shape[n_l,n_t], loc=0, scale=1/scale[n_l,n_t])
    
    pdf = pdf/np.sum(pdf, axis=(1,2))[:,np.newaxis,np.newaxis]
    
    # PDF
    if rank == 0:
        if not os.path.isdir(directory + '/plots'):
            os.mkdir(directory + '/plots')
    
        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
        for n_l in range(N_layer):
            for n_t in range(N_type):
                ax.plot(r, pdf[n_l,n_t,:], linewidth=1, color='black', label='Layer ' + str(n_l) + ' Type ' + str(n_t))
        ax.set_xlabel('Particle Radius (nm)')
        ax.set_ylabel('PDF')
        ax.legend()
        plt.savefig(directory + '/plots/PDF')
        plt.close()
    
    return pdf, r
