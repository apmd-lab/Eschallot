import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.mie.simulate_particle as sim
import eschallot.util.read_mat_data as rmd
import eschallot.montecarlo.monte_carlo_BSDF as mc
import time
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

# Simulation Settings
pht_per_wvl = 5e6 # number of photons (on average) to simulate for each sampled wavelength
lam_step = 10 # wavelength sampling step size (in nm)
lam1 = 450 # wavelength start (nm)
lam2 = 450 # wavelength end (nm)
polarization = 'x' # 'random', 'x', or 'y'

wvl = int((lam2 - lam1)/lam_step) + 1
wavelength = np.linspace(lam1, lam2, wvl, endpoint=True)
n_photon = wvl*pht_per_wvl

antireflective = 0 # assume antireflective coating on the top surface
Lambertian_sub = 0 # assume the substrate is a Lambertian scatterer
perfect_absorber = 0 # assume the substrate is a perfect absorber

wvl_for_polar_plots = np.array([450]) # wavelengths at which the polar reflection/transmission plots are drawn

output_filename = 'directional_reflector'
subgroup = 64 # head + procs (total number of cores allocated for this job must be a multiple of this number)

N_theta_BSDF = 121
N_phi_BSDF = 18
init_theta = 0*np.pi/180
init_phi = 0*np.pi/180

# Film Configuration
config_layer = np.array(['Air','PMMA','Air']) # from top(incident side) to bottom(exit side), including background media
layer_thickness = 100e3 # thickness of the particle-dispersed layer (nm)
f_vol = 0.05 # particle volume fraction

# Define Phase Function Computation Grid
nu = np.linspace(0, 1, 501) # for even area spacing along theta
theta_pf = np.flip(np.arccos(2*nu[1:-1]-1))
phi_pf = np.linspace(0, 2*np.pi, 180, endpoint=False)

# Set (or load) Particle Geometry
config = np.array(['TiO2_Sarkar'] + ['SiO2_bulk','TiO2_Sarkar']*13) # from out to in
r_profile = np.array([1914.6,
                      1855.45,
                      1831.13,
                      1762.56,
                      1641.87,
                      1565.25,
                      1529.71,
                      1470.91,
                      1384.57,
                      1338.21,
                      1324.17,
                      1226.17,
                      1201.5,
                      1134.08,
                      1106.81,
                      940.336,
                      915.156,
                      904.614,
                      883.581,
                      560.566,
                      529.728,
                      464.897,
                      440.12,
                      347.349,
                      326.115,
                      290.854,
                      213.994,
                      ])

density = f_vol/((4*np.pi*r_profile[0]**3)/3)

# Compute Mie Scattering Quantities
mat_profile = np.hstack((config_layer[1], config))
mat_type = list(set(mat_profile))
mat_data_dir = None

raw_wavelength, mat_dict_default = rmd.load_all(wavelength, 'n_k', mat_type)
if mat_data_dir is not None:
    raw_wavelength, mat_dict_custom = rmd.load_all(wavelength, 'n_k', mat_type, directory=mat_data_dir)
else:
    mat_dict_custom = dict()
mat_dict = {**mat_dict_default, **mat_dict_custom}
    
n = np.zeros((wvl, mat_profile.size)).astype(complex)
count = 0
for mat in mat_profile:
    n[:,count] = mat_dict[mat]
    count += 1

Qs, Qa, Qe, pf, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
    S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(wavelength, theta_pf, phi_pf, r_profile, n)
Qa[Qa < 0] = 0
phase_fct = np.zeros((2, wvl, theta_pf.size, phi_pf.size))
phase_fct[0,:,:,:] = pf.copy()
phase_fct[1,:,:,:] = np.roll(pf, int(phi_pf.size/4), axis=2)
    
C_sca = (np.pi*r_profile[0]**2*Qs).flatten()
C_abs = (np.pi*r_profile[0]**2*Qa).flatten()

if rank == 0:
    if not os.path.isdir(directory + '/data'):
        os.mkdir(directory + '/data')
    np.savez(directory + '/data/Mie_data_' + output_filename, C_sca=C_sca, C_abs=C_abs, phase_fct=phase_fct)

# Sync All Processes
comm.Barrier()

# Run the Monte Carlo Simulation
simulation = mc.monte_carlo(wavelength,
                            theta_pf,
                            phi_pf,
                            N_theta_BSDF,
                            N_phi_BSDF,
                            layer_thickness,
                            config_layer,
                            density,
                            C_sca,
                            C_abs,
                            phase_fct,
                            antireflective=antireflective,
                            Lambertian_sub=Lambertian_sub,
                            perfect_absorber=perfect_absorber,
                            init_theta=init_theta,
                            init_phi=init_phi,
                            polarization=polarization,
                            mat_data_dir=mat_data_dir,
                            wvl_for_polar_plots=wvl_for_polar_plots,
                            )

t1 = time.time()
simulation.run_simulation(directory, comm, size, rank, status, n_photon, output_filename, subgroup)
t2 = time.time()
if rank == 0:
    print('Simlation Time: ' + str(t2 - t1) + ' s', flush=True)

    simulation.compute_BSDF(directory, size, output_filename, subgroup)