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

with np.load(directory[:-9] + '/data/sweep_parameters.npz') as param_filename:
    thickness = param_filename['thickness']
    f_vol_tot = param_filename['f_vol_tot']
    mat_particle = param_filename['mat_particle']
    r_particle = param_filename['r_particle']
    #f_num = param_filename['f_num']
    index = param_filename['index']
    wavelength = param_filename['wavelength']
    pht_per_wvl = param_filename['pht_per_wvl']
    init_theta = param_filename['init_theta']
    init_phi = param_filename['init_phi']
    nu = param_filename['nu']
    phi_in_BSDF = param_filename['phi_in_BSDF']
    phi_out_BSDF = param_filename['phi_out_BSDF']
    identifier = param_filename['identifier'][0]

# Simulation Settings
wvl = wavelength.size
n_layer = 1 # number of layers in the film
antireflective = 0 # assume antireflective coating on the top surface
Lambertian_sub = 0 # assume the substrate is a Lambertian scatterer
perfect_absorber = 0 # assume the substrate is a perfect absorber
n_particle = 1 # number of particle radii to sample (if polydisperse); set to 1 if monodisperse
isotropic = 1 # set to 1 if particle is spherical (i.e. has the same angular scattering distribution regardless of the angle of incidence)
single_scattering_approx = 1 # valid when f_vol_tot < 1% and k_ext*d_avg > 30
                             # d_avg: average distance between particles ~= length of one side of a unit cube surrounding the particle
                             #        = ((4*pi)/(3*f_vol))**(1/3)*r
                             # REF: "Conditions of applicability of the single-scattering approximation" (2007)

n_photon = wvl*pht_per_wvl
wvl_for_polar_plots = wavelength.copy() # wavelengths at which the polar reflection/transmission plots are drawn
angle_for_spectral_plots = np.array([[30,0]])*np.pi/180 # angles (polar, azimuthal) at which the spectral reflection/transmission plots are drawn
polarization = 'random' # 'random', 'x', or 'y'

# Film Configuration
config_layer = np.array(['Air','PMMA','PMMA']) # from top(incident side) to bottom(exit side), including background media
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
layer_thickness = np.array([thickness]) # thickness of each layer (nm)

# Phase Function (differential cross section) Computation Settings
theta_inc = np.array([0])
phi_inc = np.array([0])
phi_out = np.linspace(0, 2*np.pi, 360, endpoint=False)
nu_pf = np.linspace(0, 1, 5001) # for even area spacing along theta
theta_out = np.flip(np.arccos(2*nu_pf[1:-1]-1))

# Set Particle Dispersion Quantities
#pdf, r_list = pd.gamma_distribution(n_particle, f_num, r_mean*scale, scale)
#volume_pdf = pdf*r_list[np.newaxis,np.newaxis,:]**3 # N_layer x N_type x n_particle
#f_vol = f_vol_tot[:,np.newaxis,np.newaxis]*volume_pdf/np.sum(volume_pdf, axis=(1, 2))[:,np.newaxis,np.newaxis]
f_vol = f_vol_tot[:,np.newaxis,np.newaxis] #*f_num[np.newaxis,:,np.newaxis] # N_layer x N_type x n_particle
n_type = f_vol.shape[1]

# Set Particle Geometry
config = {} # index: particle type; from out to in
r_profile = {}
for n_t in range(n_type):
    for p in range(n_particle):
        if n_t == 0:
            config[n_t,p] = mat_particle # set the material of each particle type
            r_profile[n_t,p] = r_particle
outer_radius = np.array([[r_particle[0]]])

density = np.zeros((n_layer, n_type, n_particle))
for l in range(n_layer):
    V_eff = np.sum(f_vol[l,:,:]*(4*np.pi*outer_radius**3)/3)
    if V_eff > 0:
        density[l,:,:] = f_vol[l,:,:]/V_eff

# Check Validity of the Single Scattering Approximation (if applicable)
#if single_scattering_approx:
#    for l in range(n_layer):
#        assert np.sum(f_vol[l,:,:]) < 0.01
#        
#        if np.sum(f_vol[l,:,:]) != 0:
#            assert ((4*np.pi)/(3*np.sum(f_vol[l,:,:])))**(1/3)*np.min(outer_radius) > 30

# Compute Mie Scattering Quantities
quo, rem = divmod(n_particle, size)
data_size = np.array([quo + 1 if p < rem else quo for p in range(size)])
data_disp = np.array([sum(data_size[:p]) for p in range(size+1)])
f_vol_proc = f_vol[:,:,data_disp[rank]:data_disp[rank+1]]
config_proc = {}
r_profile_proc = {}
for n_t in range(n_type):
    for p in range(data_size[rank]):
        config_proc[n_t,p] = config[n_t,data_disp[rank]+p]
        r_profile_proc[n_t,p] = r_profile[n_t,data_disp[rank]+p]
outer_radius_proc = outer_radius[:,data_disp[rank]:data_disp[rank+1]]

# Load Material Refractive Index
mat_profile = np.hstack((config_layer, mat_particle))
mat_type = list(set(mat_profile))
raw_wavelength, mat_dict = rmd.load_all(wavelength, 'n_k', mat_type)

C_sca_surf_proc = np.zeros((n_type, data_size[rank], wvl))
C_abs_surf_proc = np.zeros((n_type, data_size[rank], wvl))
C_sca_bulk_proc = np.zeros((n_layer, n_type, data_size[rank], wvl))
C_abs_bulk_proc = np.zeros((n_layer, n_type, data_size[rank], wvl))
diff_scat_CS_proc = np.zeros((n_layer, n_type, data_size[rank], 2, wvl, theta_out.size, phi_out.size))

for l in range(n_layer+1):
    if l != 0 and not single_scattering_approx:
        mat_ema = np.hstack((mat_particle, config_layer[l]))
        n_ema = np.zeros((wvl, mat_ema.size)).astype(complex)
        count = 0
        for mat in mat_ema:
            n_ema[:,count] = mat_dict[mat]
            count += 1
        f_vol_ema = np.sum(f_vol[l-1,:,:], axis=1)
        f_vol_ema = np.append(f_vol_ema, 1-f_vol_tot[l-1])

    for n_t in range(n_type):
        for p in range(data_size[rank]):
            mat_profile = np.hstack((config_layer[l], config_proc[n_t,p]))
            n = np.zeros((wvl, mat_profile.size)).astype(complex)
            count = 0
            for mat in mat_profile:
                n[:,count] = mat_dict[mat]
                count += 1
                
            if l != 0 and not single_scattering_approx:
                eps_eff = ema.multitype_particle_medium(wavelength, n_ema, f_vol_ema)
                n[:,0] = np.sqrt(eps_eff)
        
            Qs, Qa, Qe, pf, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(wavelength, theta_out, phi_out, r_profile_proc[n_t,p], n)
            Qa[Qa < 0] = 0
    
            if l == 0:
                C_sca_surf_proc[n_t,p,:] = (np.pi*outer_radius_proc[n_t,p]**2*Qs).flatten()
                C_abs_surf_proc[n_t,p,:] = (np.pi*outer_radius_proc[n_t,p]**2*Qa).flatten()
            else:
                C_sca_bulk_proc[l-1,n_t,p,:] = (np.pi*outer_radius_proc[n_t,p]**2*Qs).flatten()
                C_abs_bulk_proc[l-1,n_t,p,:] = (np.pi*outer_radius_proc[n_t,p]**2*Qa).flatten()
                if not single_scattering_approx:
                    str_fact = strf.structure_factor(wavelength, np.array([0]), np.array([0]), theta_out, phi_out, f_vol_proc[l-1,n_t,p], n[:,0], np.array([outer_radius_proc[n_t,p]]), 1)
                    diff_scat_CS_proc[l-1,n_t,p,0,:,:,:] = pf*str_fact[:,0,0,:,:]
                else:
                    diff_scat_CS_proc[l-1,n_t,p,0,:,:,:] = pf

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
C_abs_bulk = C_abs_bulk_temp.reshape(n_particle, n_layer, n_type, wvl).transpose(1,2,0,3)

data_size_temp = data_size*n_layer*n_type*2*wvl*theta_out.size*phi_out.size
data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(size)]).astype(np.float64)

diff_scat_CS_temp = np.zeros(n_layer*n_type*n_particle*2*wvl*theta_out.size*phi_out.size)
comm.Allgatherv(diff_scat_CS_proc.transpose(2,0,1,3,4,5,6).reshape(-1), [diff_scat_CS_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
diff_scat_CS = diff_scat_CS_temp.reshape(n_particle, n_layer, n_type, 2, wvl, theta_out.size, phi_out.size).transpose(1,2,0,3,4,5,6)

diff_scat_CS[:,:,:,1,:,:,:] = np.roll(diff_scat_CS[:,:,:,0,:,:,:], int(phi_out.size/4), axis=5)

subgroup = 8 # head + procs (total number of cores allocated for this job must be a multiple of this number)

if rank == 0:
    np.savez(directory[:-9] + '/data/Mie_data_' + identifier, C_sca_surf=C_sca_surf, C_abs_surf=C_abs_surf, C_sca_bulk=C_sca_bulk, C_abs_bulk=C_abs_bulk,
             diff_scat_CS=diff_scat_CS)

# Sync All Processes
comm.Barrier()

# Set Incidence Angles and Scattered Angle Resolutions
theta_temp = np.flip(np.arccos(2*nu-1))
theta_in_BSDF = theta_temp[theta_temp <= np.pi/2]
theta_out_BRDF_edge = theta_temp[theta_temp >= np.pi/2]
theta_out_BRDF_center = (theta_out_BRDF_edge[:-1] + theta_out_BRDF_edge[1:])/2
theta_out_BRDF_center = np.insert(theta_out_BRDF_center, 0, np.pi/2)
theta_out_BRDF_center = np.append(theta_out_BRDF_center, np.pi)
theta_out_BTDF_edge = theta_temp[theta_temp <= np.pi/2]
theta_out_BTDF_center = (theta_out_BTDF_edge[:-1] + theta_out_BTDF_edge[1:])/2
theta_out_BTDF_center = np.insert(theta_out_BTDF_center, 0, 0)
theta_out_BTDF_center = np.append(theta_out_BTDF_center, np.pi/2)
if rank == 0:
    BRDF_spec = np.zeros((wvl, init_theta.size, init_phi.size, theta_out_BRDF_center.size, phi_out_BSDF.size))
    BRDF_diff = np.zeros((wvl, init_theta.size, init_phi.size, theta_out_BRDF_center.size, phi_out_BSDF.size))
    BTDF_ball = np.zeros((wvl, init_theta.size, init_phi.size, theta_out_BTDF_center.size, phi_out_BSDF.size))
    BTDF_diff = np.zeros((wvl, init_theta.size, init_phi.size, theta_out_BTDF_center.size, phi_out_BSDF.size))
for th_in in range(init_theta.size):
    if init_theta[th_in] == np.pi/2 and rank == 0:
        BRDF_spec[:,th_in,0,0] = 1
    else:
        for ph_in in range(init_phi.size):
            t1 = time.time()
            mc.monte_carlo(wavelength, theta_inc, theta_out, phi_inc, phi_out, theta_in_BSDF, phi_in_BSDF, theta_out_BRDF_edge, theta_out_BRDF_center,
                           theta_out_BTDF_edge, theta_out_BTDF_center, phi_out_BSDF, wvl_for_polar_plots, angle_for_spectral_plots,
                           layer_thickness, RI, density, C_sca_surf, C_abs_surf, C_sca_bulk, C_abs_bulk, diff_scat_CS,
                           fine_roughness, coarse_roughness,
                           antireflective=antireflective, Lambertian_sub=Lambertian_sub, perfect_absorber=perfect_absorber,
                           isotropic=isotropic, init_theta=init_theta[th_in], init_phi=init_phi[ph_in], polarization=polarization).normal_hemispherical(directory[:-9], comm, size, rank,
                                                                                                                                                        status, n_photon, identifier,
                                                                                                                                                        subgroup)
            t2 = time.time()
            comm.Barrier()
            if rank == 0:
                I_tot = np.zeros(wvl)
                R_spec_tot = np.zeros(wvl)
                R_diff_tot = np.zeros(wvl)
                R_scat_tot = np.zeros(wvl)
                T_ball_tot = np.zeros(wvl)
                T_diff_tot = np.zeros(wvl)
                T_scat_tot = np.zeros(wvl)
                A_medium_tot = np.zeros(wvl)
                A_particle_tot = np.zeros(wvl)
                A_TIR_tot = np.zeros(wvl)
                
                inc_angle_tot = np.zeros((theta_in_BSDF.size, phi_in_BSDF.size, wvl))
                reflect_angle_spec_tot = np.zeros((theta_out_BRDF_center.size, phi_out_BSDF.size, wvl))
                reflect_angle_diff_tot = np.zeros((theta_out_BRDF_center.size, phi_out_BSDF.size, wvl))
                transmit_angle_ball_tot = np.zeros((theta_out_BTDF_center.size, phi_out_BSDF.size, wvl))
                transmit_angle_diff_tot = np.zeros((theta_out_BTDF_center.size, phi_out_BSDF.size, wvl))
                for n in range(size//subgroup):
                    data = np.load(directory[:-9] + "/data/" + identifier + "_MC_" + str(int(n*subgroup)) +".npz")
                    I_tot = I_tot + data['I']
                    R_spec_tot = R_spec_tot + data['R_spec']
                    R_diff_tot = R_diff_tot + data['R_diff']
                    R_scat_tot = R_scat_tot + data['R_scat']
                    T_ball_tot = T_ball_tot + data['T_ball']
                    T_diff_tot = T_diff_tot + data['T_diff']
                    T_scat_tot = T_scat_tot + data['T_scat']
                    A_medium_tot = A_medium_tot + data['A_medium']
                    A_particle_tot = A_particle_tot + data['A_particle']
                    A_TIR_tot = A_TIR_tot + data['A_TIR']
                    
                    inc_angle_tot = inc_angle_tot + data['inc_angle']
                    reflect_angle_spec_tot = reflect_angle_spec_tot + data['reflect_angle_spec']
                    reflect_angle_diff_tot = reflect_angle_diff_tot + data['reflect_angle_diff']
                    transmit_angle_ball_tot = transmit_angle_ball_tot + data['transmit_angle_ball']
                    transmit_angle_diff_tot = transmit_angle_diff_tot + data['transmit_angle_diff']
                
                R_spec_tot /= I_tot
                R_diff_tot /= I_tot
                R_scat_tot /= I_tot
                T_ball_tot /= I_tot
                T_diff_tot /= I_tot
                T_scat_tot /= I_tot
                A_medium_tot /= I_tot
                A_particle_tot /= I_tot
                A_TIR_tot /= I_tot
                
                for w in range(wvl):
                    BRDF_spec[w,th_in,ph_in,:,:] = reflect_angle_spec_tot[:,:,w]/I_tot[w]
                    BRDF_diff[w,th_in,ph_in,:,:] = reflect_angle_diff_tot[:,:,w]/I_tot[w]
                    BTDF_ball[w,th_in,ph_in,:,:] = transmit_angle_ball_tot[:,:,w]/I_tot[w]
                    BTDF_diff[w,th_in,ph_in,:,:] = transmit_angle_diff_tot[:,:,w]/I_tot[w]

if rank == 0:
    np.savez(directory[:-9] + "/data/param_sweep_" + identifier + "_BSDF" + str(index), BRDF_spec=BRDF_spec, BRDF_diff=BRDF_diff, BTDF_ball=BTDF_ball, BTDF_diff=BTDF_diff,
             wavelength=wavelength, theta_in_BSDF=theta_in_BSDF, theta_out_BRDF=theta_out_BRDF_center, theta_out_BTDF=theta_out_BTDF_center, phi_out_BSDF=phi_out_BSDF,
             f_vol=f_vol, R_spec=R_spec_tot, R_diff=R_diff_tot, R_scat=R_scat_tot, T_ball=T_ball_tot, T_diff=T_diff_tot, T_scat=T_scat_tot, A_medium=A_medium_tot,
             A_particle=A_particle_tot, A_TIR=A_TIR_tot, theta_out_BRDF_edge=theta_out_BRDF_edge, theta_out_BTDF_edge=theta_out_BTDF_edge)
