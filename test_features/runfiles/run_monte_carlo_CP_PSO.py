import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
import mie.simulate_particle as sim
import util.read_mat_data as rmd
import montecarlo.effective_medium_approximation as ema
import montecarlo.monte_carlo_BSDF_anisotropic as mc
import montecarlo.structure_factor as strf
import montecarlo.particle_distribution as pd
import time
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

with np.load(directory + '/data/CP_PSO_optimization_parameters.npz') as param_filename:
    param_layer_thickness = param_filename['layer_thickness']
    param_f_vol_tot = param_filename['f_vol_tot']
    param_f_num1 = param_filename['f_num1']
    param_r_mean1 = param_filename['r_mean1']
    param_r_mean2 = param_filename['r_mean2']
    param_scale1 = param_filename['scale1']
    param_scale2 = param_filename['scale2']
    param_mat1 = param_filename['mat1']
    param_mat2 = param_filename['mat2']
    
    pht_per_wvl = param_filename['pht_per_wvl']
    lam_step = param_filename['lam_step']
    lam1 = param_filename['lam1']
    lam2 = param_filename['lam2']

# Simulation Settings
wvl = int((lam2 - lam1)/lam_step) + 1
n_layer = 1 # number of layers in the film
antireflective = 0 # assume antireflective coating on the top surface
Lambertian_sub = 0 # assume the substrate is a Lambertian scatterer
perfect_absorber = 0 # assume the substrate is a perfect absorber
n_particle = 100 # number of particle radii to sample (if polydisperse); set to 1 if monodisperse
isotropic = 1 # set to 1 if particle is spherical (i.e. has the same angular scattering distribution regardless of the angle of incidence)

wavelength = np.linspace(lam1, lam2, wvl, endpoint=True)
n_photon = wvl*pht_per_wvl
wvl_for_polar_plots = np.array([550]) # wavelengths at which the polar reflection/transmission plots are drawn
angle_for_spectral_plots = np.array([[0,0]])*np.pi/180 # angles (azimuthal, polar) at which the spectral reflection/transmission plots are drawn
polarization = 'random' # 'random', 'x', or 'y'

# Film Configuration
config_layer = np.array(['Air','PMMA','Air']) # from top(incident side) to bottom(exit side), including background media
mat_type = list(set(config_layer))
raw_wavelength, mat_dict = rmd.load_all(wavelength, 'n_k', mat_type)
RI = np.zeros((n_layer+2, wvl)).astype(complex)
count = 0
for mat in config_layer:
    RI[count,:] = mat_dict[mat]
    count += 1

# Refer to Hwang et al. "Designing angle-independent structural colors using Monte Carlo simulations of multiple scattering", PNAS (2021).
fine_roughness = 0
coarse_roughness = 0
layer_thickness = np.array([param_layer_thickness]) # thickness of each layer (nm)

# Phase Function (differential cross section) Computation Settings
theta_inc = np.array([0])
phi_inc = np.array([0])
phi_out = np.linspace(0, 2*np.pi, 180, endpoint=False)
nu = np.linspace(0, 1, 201) # for even area spacing along theta
theta_out = np.flip(np.arccos(2*nu[1:-1]-1))

# Set Particle Dispersion Quantities
f_vol_tot = np.array([param_f_vol_tot]) # total particle volume fraction for each layer
f_num = np.array([[param_f_num1,1-param_f_num1]]) # relative number densities of different particle types (should add up to 1; dimension -> n_layer x n_type; n_type=1 for unimodal, 2 for bimodal, etc.)
r_mean = np.array([[param_r_mean1,param_r_mean2]]) # mean radii (dimension -> n_layer x n_type)
scale = np.array([[param_scale1,param_scale2]]) # determines the spread of the gamma distribution (smaller scale -> larger spread)
pdf, r_list = pd.gamma_distribution(n_particle, f_num, r_mean*scale, scale)
f_vol = f_vol_tot[:,np.newaxis,np.newaxis]*pdf
n_type = f_vol.shape[1]

# Set Particle Geometry
mat_list = np.array(['TiO2_Sarkar','SiO2_bulk','ZrO2_li','BaSO4_li'])
config = {} # index: particle type; from out to in
r_profile = {}
for n_t in range(n_type):
    for p in range(n_particle):
        if n_t == 0:
            config[n_t,p] = np.array([mat_list[mat1]]) # set the material of particle type 1
        elif n_t == 1:
            config[n_t,p] = np.array([mat_list[mat2]]) # set the material of particle type 2
        r_profile[n_t,p] = np.array([r_list[p]])
outer_radius = r_list.copy()

density = np.zeros((n_layer, n_type, n_particle))
for p in range(n_particle):
    density[:,:,p] = f_vol[:,:,p]/((4*np.pi*outer_radius[p]**3)/3)

# Compute Mie Scattering Quantities
quo, rem = divmod(r_list.size, size)
data_size = np.array([quo + 1 if p < rem else quo for p in range(size)])
data_disp = np.array([sum(data_size[:p]) for p in range(size+1)])
f_vol_proc = f_vol[:,data_disp[rank]:data_disp[rank+1]]
config_proc = {}
r_profile_proc = {}
for n_t in range(n_type):
    for p in range(data_size[rank]):
        config_proc[n_t,p] = config[n_t,data_disp[rank]+p]
        r_profile_proc[n_t,p] = r_profile[n_t,data_disp[rank]+p]
outer_radius_proc = outer_radius[data_disp[rank]:data_disp[rank+1]]

C_sca_surf_proc = np.zeros((n_type, data_size[rank], wvl))
C_abs_surf_proc = np.zeros((n_type, data_size[rank], wvl))
C_sca_bulk_proc = np.zeros((n_layer, n_type, data_size[rank], wvl))
C_abs_bulk_proc = np.zeros((n_layer, n_type, data_size[rank], wvl))
diff_scat_CS_proc = np.zeros((n_layer, n_type, data_size[rank], 2, wvl, theta_out.size, phi_out.size))
for l in range(n_layer+1):
    for n_t in range(n_type):
        for p in range(data_size[rank]):
            # Create n
            mat_profile = np.hstack((config_layer[l], config_proc[n_t,p]))
            mat_type = list(set(mat_profile))
            raw_wavelength, mat_dict = rmd.load_all(wavelength, 'n_k', mat_type)
    
            n = np.zeros((wvl, mat_profile.size)).astype(complex)
            count = 0
            for mat in mat_profile:
                n[:,count] = mat_dict[mat]
                count += 1
                
            if l != 0:
                eps_eff = ema.particle_medium(n, r_profile_proc[n_t,p], r_profile_proc[n_t,p].size, np.sum(f_vol[l-1,n_t,:]))
                n[:,0] = np.sqrt(eps_eff)
        
            Qs, Qa, Qe, pf, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(wavelength, theta_out, phi_out, r_profile_proc[n_t,p], n)
            Qa[Qa < 0] = 0
    
            if l == 0:
                C_sca_surf_proc[n_t,p,:] = (np.pi*outer_radius_proc[p]**2*Qs).flatten()
                C_abs_surf_proc[n_t,p,:] = (np.pi*outer_radius_proc[p]**2*Qa).flatten()
            else:
                C_sca_bulk_proc[l-1,n_t,p,:] = (np.pi*outer_radius_proc[p]**2*Qs).flatten()
                C_abs_bulk_proc[l-1,n_t,p,:] = (np.pi*outer_radius_proc[p]**2*Qa).flatten()
                str_fact = strf.structure_factor(wavelength, np.array([0]), np.array([0]), theta_out, phi_out, f_vol_proc[l-1,n_t,p], n[:,0], np.array([outer_radius_proc[p]]), 1)
                diff_scat_CS_proc[l-1,n_t,p,0,:,:,:] = pf*str_fact[:,0,0,:,:]

data_size_temp = data_size*n_type*wvl
data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(size)]).astype(np.float64)

C_sca_surf_temp = np.zeros(n_type*n_particle*wvl)
comm.Allgatherv(C_sca_surf_proc.transpose(1,0,2).reshape(-1), [C_sca_surf_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
C_sca_surf = C_sca_surf_temp.reshape(n_particle, n_type, wvl).transpose(1,0,2)

C_abs_surf_temp = np.zeros(n_type*n_particle*wvl)
comm.Allgatherv(C_abs_surf_proc.transpose(1,0,2).reshape(-1), [C_abs_surf_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
C_abs_surf = C_abs_surf_temp.reshape(n_particle, n_type, wvl).transpose(1,0,2)

data_size_temp = data_size*n_layer*n_type*wvl
data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(size)]).astype(np.float64)

C_sca_bulk_temp = np.zeros(n_layer*n_type*n_particle*wvl)
comm.Allgatherv(C_sca_bulk_proc.transpose(2,0,1,3).reshape(-1), [C_sca_bulk_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
C_sca_bulk = C_sca_bulk_temp.reshape(n_particle, n_layer, n_type, wvl).transpose(1,2,0,3)

C_abs_bulk_temp = np.zeros(n_layer*n_type*n_particle*wvl)
comm.Allgatherv(C_abs_bulk_proc.transpose(2,0,1,3).reshape(-1), [C_abs_bulk_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
C_abs_bulk = C_abs_bulk_temp.reshape(n_particle, n_layer, wvl).transpose(1,2,0,3)

data_size_temp = data_size*n_layer*n_type*2*wvl*theta_out.size*phi_out.size
data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(size)]).astype(np.float64)

diff_scat_CS_temp = np.zeros(n_layer*n_type*n_particle*2*wvl*theta_out.size*phi_out.size)
comm.Allgatherv(diff_scat_CS_proc.transpose(2,0,1,3,4,5,6).reshape(-1), [diff_scat_CS_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
diff_scat_CS = diff_scat_CS_temp.reshape(n_particle, n_layer, n_type, 2, wvl, theta_out.size, phi_out.size).transpose(1,2,0,3,4,5,6)

diff_scat_CS[:,:,:,1,:,:,:] = np.roll(diff_scat_CS[:,:,:,0,:,:,:], int(phi_out.size/4), axis=5)

identifier = 'CP_PSO'
subgroup = 6 # head + procs (total number of cores allocated for this job must be a multiple of this number)

if rank == 0:
    if not os.path.isdir(directory + '/data'):
        os.mkdir(directory + '/data')
    np.savez(directory + '/data/Mie_data' + identifier, C_sca_surf=C_sca_surf, C_abs_surf=C_abs_surf, C_sca_bulk=C_sca_bulk, C_abs_bulk=C_abs_bulk,
             diff_scat_CS=diff_scat_CS)

# Sync All Processes
comm.Barrier()

# Set Incidence Angles and Scattered Angle Resolutions
theta_in_BSDF = np.linspace(0, 90, 2, endpoint=True)*np.pi/180
nu = np.linspace(0, 1, 21) # for even area spacing along theta (number of angles must be odd to always include pi/2)
theta_temp = np.flip(np.arccos(2*nu-1))
theta_out_BRDF_edge = theta_temp[theta_temp >= np.pi/2]
theta_out_BRDF_center = (theta_out_BRDF_edge[:-1] + theta_out_BRDF_edge[1:])/2
theta_out_BRDF_center = np.insert(theta_out_BRDF_center, 0, np.pi/2)
theta_out_BRDF_center = np.append(theta_out_BRDF_center, np.pi)
theta_out_BTDF_edge = theta_temp[theta_temp <= np.pi/2]
theta_out_BTDF_center = (theta_out_BTDF_edge[:-1] + theta_out_BTDF_edge[1:])/2
theta_out_BTDF_center = np.insert(theta_out_BTDF_center, 0, 0)
theta_out_BTDF_center = np.append(theta_out_BTDF_center, np.pi/2)
phi_out_BSDF = np.linspace(0, 2*np.pi, 18, endpoint=False) # set azimuthal angle sampling points
if rank == 0:
    BRDF_spec = np.zeros((wvl, theta_in_BSDF.size, theta_out_BRDF_center.size, phi_out_BSDF.size))
    BRDF_diff = np.zeros((wvl, theta_in_BSDF.size, theta_out_BRDF_center.size, phi_out_BSDF.size))
    BTDF_ball = np.zeros((wvl, theta_in_BSDF.size, theta_out_BTDF_center.size, phi_out_BSDF.size))
    BTDF_diff = np.zeros((wvl, theta_in_BSDF.size, theta_out_BTDF_center.size, phi_out_BSDF.size))
for th_in in range(theta_in_BSDF.size):
    if theta_in_BSDF[th_in] == np.pi/2 and rank == 0:
        BRDF_spec[:,th_in,0,0] = 1
    else:
        t1 = time.time()
        mc.monte_carlo(wavelength, theta_inc, theta_out, phi_inc, phi_out, theta_in_BSDF, theta_out_BRDF_edge, theta_out_BRDF_center,
                       theta_out_BTDF_edge, theta_out_BTDF_center, phi_out_BSDF, wvl_for_polar_plots, angle_for_spectral_plots,
                       layer_thickness, RI, density, C_sca_surf, C_abs_surf, C_sca_bulk, C_abs_bulk, diff_scat_CS,
                       fine_roughness, coarse_roughness,
                       antireflective=antireflective, Lambertian_sub=Lambertian_sub, perfect_absorber=perfect_absorber,
                       isotropic=isotropic, init_angle=theta_in_BSDF[th_in], polarization=polarization).normal_hemispherical(directory, comm, size, rank,
                                                                                                                             status, n_photon, identifier,
                                                                                                                             subgroup)
        t2 = time.time()
        comm.Barrier()
        if rank == 0:
            I_tot = np.zeros(wvl)
            R_spec_tot = np.zeros(wvl)
            R_diff_tot = np.zeros(wvl)
            T_ball_tot = np.zeros(wvl)
            T_diff_tot = np.zeros(wvl)
            A_medium_tot = np.zeros(wvl)
            A_particle_tot = np.zeros(wvl)
            A_TIR_tot = np.zeros(wvl)
            A_scat_tot = np.zeros(wvl)
            
            inc_angle_tot = np.zeros((theta_in_BSDF.size, wvl))
            reflect_angle_spec_tot = np.zeros((theta_out_BRDF_center.size, phi_out_BSDF.size, wvl))
            reflect_angle_diff_tot = np.zeros((theta_out_BRDF_center.size, phi_out_BSDF.size, wvl))
            transmit_angle_ball_tot = np.zeros((theta_out_BTDF_center.size, phi_out_BSDF.size, wvl))
            transmit_angle_diff_tot = np.zeros((theta_out_BTDF_center.size, phi_out_BSDF.size, wvl))
            for n in range(size//subgroup):
                data = np.load(directory + "/data/" + identifier + "_MC_" + str(int(n*subgroup)) +".npz")
                I_tot = I_tot + data['I']
                R_spec_tot = R_spec_tot + data['R_spec']
                R_diff_tot = R_diff_tot + data['R_diff']
                T_ball_tot = T_ball_tot + data['T_ball']
                T_diff_tot = T_diff_tot + data['T_diff']
                A_medium_tot = A_medium_tot + data['A_medium']
                A_particle_tot = A_particle_tot + data['A_particle']
                A_TIR_tot = A_TIR_tot + data['A_TIR']
                A_scat_tot = A_scat_tot + data['A_scat']
                
                inc_angle_tot = inc_angle_tot + data['inc_angle']
                reflect_angle_spec_tot = reflect_angle_spec_tot + data['reflect_angle_spec']
                reflect_angle_diff_tot = reflect_angle_diff_tot + data['reflect_angle_diff']
                transmit_angle_ball_tot = transmit_angle_ball_tot + data['transmit_angle_ball']
                transmit_angle_diff_tot = transmit_angle_diff_tot + data['transmit_angle_diff']
            
            for w in range(wvl):
                BRDF_spec[w,th_in,:,:] = reflect_angle_spec_tot[:,:,w]/I_tot[w]
                BRDF_diff[w,th_in,:,:] = reflect_angle_diff_tot[:,:,w]/I_tot[w]
                BTDF_ball[w,th_in,:,:] = transmit_angle_ball_tot[:,:,w]/I_tot[w]
                BTDF_diff[w,th_in,:,:] = transmit_angle_diff_tot[:,:,w]/I_tot[w]

if rank == 0:
    if not os.path.isdir(directory + '/results'):
        os.mkdir(directory + '/results')
    np.savez(directory + "/results/" + identifier + "_BRDF", BRDF_spec=BRDF_spec, BRDF_diff=BRDF_diff, BTDF_ball=BTDF_ball, BTDF_diff=BTDF_diff,
             wavelength=wavelength, theta_in_BSDF=theta_in_BSDF, theta_out_BRDF=theta_out_BRDF_center, theta_out_BTDF=theta_out_BTDF_center, phi_out_BSDF=phi_out_BSDF,
             f_vol=f_vol, r_list=r_list)
