import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
from itertools import product
import time
import subprocess

thickness = np.array([1000000])
# Note: thickness at which x% of photons scatter at least once (the number is the extinction cross section) -> -np.log(1-x)*137891.8083528611
f_vol_tot = np.array([0.1])
init_theta = np.linspace(40, 70, 7)*np.pi/180

wvl = 1
n_theta = 121
n_phi = 18
wavelength = np.linspace(450, 460, wvl) # in nm
init_phi = np.array([0])*np.pi/180
nu = np.linspace(0, 1, n_theta) # for even area spacing along theta (number of angles must be odd to always include pi/2)
phi_in_BSDF = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
phi_out_BSDF = np.linspace(0, 2*np.pi, n_phi, endpoint=False) # set azimuthal angle sampling points
pht_per_wvl = 1e7 # number of photons (on average) to simulate for each sampled wavelength

data_sweep = np.array(list(product(thickness, f_vol_tot, init_theta)))
pht_ratio = np.array(list(product(thickness/np.min(thickness), f_vol_tot/np.min(f_vol_tot), init_theta)))
pht_per_wvl = pht_per_wvl/(pht_ratio[:,0]*pht_ratio[:,1])
n_core = 288

identifier = np.array(['theta148_phi0_lambda450_rmax3000_dtheta5_Bwd'])
suffix = '_directional_projector'

incidence = 'multi' # normal / oblique / multi

# Import Optimized Particle Data
with np.load(directory[:-9] + '/results/topopt_result_data_' + identifier[0] + '.npz') as opt_data:
    N_layer = int(opt_data['N_layer'][1])
    r_particle = opt_data['r'][1,:N_layer]
    n_particle = opt_data['n'][1,:,:N_layer+1]
    
    mat_particle = np.array(['']*N_layer, dtype='<U20') # particle only, from out to in
    for nl in range(N_layer):
        if np.real(n_particle[-1,nl+1]) < 1.7:
            mat_particle[nl] = 'SiO2_bulk'
        else:
            mat_particle[nl] = 'TiO2_Sarkar'

# Write Bash File for Monte Carlo Simulation
with open(directory + '/monte_carlo_batch.sh', 'w') as batch_file:
    ### For MIT Supercloud
    batch_file.write("#!/bin/bash\n\n")
    batch_file.write("#SBATCH -o /home/gridsan/smin/python_scripts/Eschallot/slurm/monte_carlo_batch.log-%j\n")
    batch_file.write("#SBATCH -n " + str(int(n_core)) + "\n\n")
    batch_file.write("source /etc/profile\n")
    batch_file.write("module load anaconda/2023a\n")
    batch_file.write("export OMP_NUM_THREADS=1\n\n")
    batch_file.write("mpirun -np " + str(int(n_core)) + " python " + directory + "/run_monte_carlo_parameter_sweep.py")

proc = subprocess.Popen(["chmod", "u+x", directory + '/monte_carlo_batch.sh']) 

# Create Folders (if missing)
if not os.path.isdir(directory[:-9] + '/results'):
    os.mkdir(directory[:-9] + '/results')
if not os.path.isdir(directory[:-9] + '/data'):
    os.mkdir(directory[:-9] + '/data')
if not os.path.isdir(directory[:-9] + '/plots'):
    os.mkdir(directory[:-9] + '/plots')
if not os.path.isdir(directory[:-9] + '/logs'):
    os.mkdir(directory[:-9] + '/logs')

# Load Previous Data (if applicable)
if os.path.exists(directory + '/results/' + 'param_sweep_' + identifier[0] + suffix + '.npz'):
    prev_data = np.load(directory + '/results/' + 'param_sweep_' + identifier[0] + suffix + '.npz')
    BRDF_spec_all = prev_data['BRDF_spec_all']
    BRDF_diff_all = prev_data['BRDF_diff_all']
    R_tgt_all = prev_data['R_tgt_all']
    R_spec_all = prev_data['R_spec_all']
    R_diff_all = prev_data['R_diff_all']
    f_tgt_all = prev_data['f_tgt_all']
    f_diff_all = prev_data['f_diff_all']
    n_d0 = prev_data['n_d0']
else:
    BRDF_spec_all = np.zeros((data_sweep.shape[0], wvl, int((n_theta-1)/2+2), n_phi))
    BRDF_diff_all = np.zeros((data_sweep.shape[0], wvl, int((n_theta-1)/2+2), n_phi))
    R_tgt_all = np.zeros((data_sweep.shape[0], wvl))
    R_spec_all = np.zeros((data_sweep.shape[0], wvl))
    R_diff_all = np.zeros((data_sweep.shape[0], wvl))
    f_tgt_all = np.zeros((data_sweep.shape[0], wvl))
    f_diff_all = np.zeros((data_sweep.shape[0], wvl))
    n_d0 = 0

# Parameter Sweep
for n_d in range(n_d0, data_sweep.shape[0]):
    thickness_sweep = data_sweep[n_d,0]
    f_vol_tot_sweep = np.array([0,data_sweep[n_d,1]])
    init_theta_sweep = np.array([data_sweep[n_d,2]])
    pht_per_wvl_sweep = pht_per_wvl[n_d]

    print('\n### Simulation ' + str(n_d), flush=True)
    print('    | Film Thickness: ' + str(thickness_sweep), flush=True)
    print('    | Total Particle Volume Fraction: ' + str(f_vol_tot_sweep[0]), flush=True)
    print('    | Angle of Incidence (polar): ' + str(init_theta_sweep[0]*180/np.pi), flush=True)
    print('    | Photons Per Wavelength: ' + str(pht_per_wvl_sweep), flush=True)

    np.savez(directory[:-9] + '/data/sweep_parameters',
             thickness=thickness_sweep,
             f_vol_tot=f_vol_tot_sweep,
             mat_particle=mat_particle,
             r_particle=r_particle,
             index=n_d,
             wavelength=wavelength,
             pht_per_wvl=pht_per_wvl_sweep,
             init_theta=init_theta_sweep,
             init_phi=init_phi,
             nu=nu,
             phi_in_BSDF=phi_in_BSDF,
             phi_out_BSDF=phi_out_BSDF,
             identifier=np.array([identifier[0] + suffix]))

    while True:
        if not os.path.exists(directory[:-9] + '/data/param_sweep_' + identifier[0]  + suffix + '_BSDF' + str(n_d) + '.npz'):
            t1 = time.time()
            proc = subprocess.Popen(["sbatch", directory + '/monte_carlo_batch.sh'])
            while True:
                if os.path.exists(directory[:-9] + '/data/param_sweep_' + identifier[0]  + suffix + '_BSDF' + str(n_d) + '.npz'):
                    break
                time.sleep(10)
            t2 = time.time()
            print('    | Simulation Time: ' + str(t2 - t1) + ' s', flush=True)
        
            queue = subprocess.Popen(["squeue"], stdout=subprocess.PIPE)
            queue_str = queue.communicate()
            jobid_temp = queue_str[0].decode('utf-8').split()
            jobid = None
            
            for i in range(len(jobid_temp)):
                ### For MIT Supercloud
                if jobid_temp[i] == 'monte_ca' and jobid_temp[i+1] == 'smin':
                    jobid = jobid_temp[i-2]
                    break
            
            if jobid is not None:
                quit_proc = subprocess.Popen(["scancel", jobid])
            
        with np.load(directory[:-9] + '/data/param_sweep_' + identifier[0]  + suffix + '_BSDF' + str(n_d) + '.npz') as load_data:
            BRDF_spec = load_data['BRDF_spec']
            BRDF_diff = load_data['BRDF_diff']
            theta_out_BRDF = load_data['theta_out_BRDF']
        
        if np.sum(np.isnan(BRDF_spec + BRDF_diff)) == 0:
            break
        else:
            os.remove(directory[:-9] + '/data/param_sweep_' + identifier[0]  + suffix + '_BSDF' + str(n_d) + '.npz')
    
    BRDF_spec_all[n_d,:,:,:] = BRDF_spec[:,0,0,:,:]
    BRDF_diff_all[n_d,:,:,:] = BRDF_diff[:,0,0,:,:]
    
    if incidence == 'normal':
        mask_theta_tgt = (theta_out_BRDF >= 140*np.pi/180)*(theta_out_BRDF <= 160*np.pi/180)
        mask_phi_tgt = (phi_out_BSDF >= 45*np.pi/180)*(phi_out_BSDF <= 135*np.pi/180) + (phi_out_BSDF >= 225*np.pi/180)*(phi_out_BSDF <= 315*np.pi/180)
    elif incidence == 'oblique':
        mask_theta_tgt = (theta_out_BRDF <= (180 - 13.3)*np.pi/180)*(theta_out_BRDF >= (180 - 33.3)*np.pi/180)
        mask_phi_tgt = (phi_out_BSDF >= 135*np.pi/180)*(phi_out_BSDF <= 225*np.pi/180)
    elif incidence == 'multi':
        mask_theta_tgt = theta_out_BRDF >= (180 - 17)*np.pi/180
        mask_phi_tgt = phi_out_BSDF > 0
        
    mask_tgt = (mask_theta_tgt[:,np.newaxis]*mask_phi_tgt[np.newaxis,:])[np.newaxis,:,:]
    R_tgt_all[n_d,:] = np.sum(BRDF_spec[:,0,0,:,:]*mask_tgt + BRDF_diff[:,0,0,:,:]*mask_tgt, axis=(1,2))
    R_spec_all[n_d,:] = np.sum(BRDF_spec[:,0,0,:,:], axis=(1,2))
    R_diff_all[n_d,:] = np.sum(BRDF_diff[:,0,0,:,:], axis=(1,2))
    f_tgt_all[n_d,:] = R_tgt_all[n_d,:]/(R_spec_all[n_d,:] + R_diff_all[n_d,:])
    f_diff_all[n_d,:] = R_tgt_all[n_d,:]/R_diff_all[n_d,:]
    
    print('\n    | Directional Reflectance (450 nm): %.2f percent' %(R_tgt_all[n_d,0]), flush=True)
    print('    | Directional Reflectance Relative to All Reflectance (450 nm): %.2f percent' %(f_tgt_all[n_d,0]), flush=True)
    print('    | Directional Reflectance Relative to Diffuse Reflectance (450 nm): %.2f percent' %(f_diff_all[n_d,0]), flush=True)

    np.savez(directory[:-9] + '/results/param_sweep_' + identifier[0] + suffix, params=data_sweep, BRDF_spec_all=BRDF_spec_all, BRDF_diff_all=BRDF_diff_all, mask_tgt=mask_tgt, R_tgt_all=R_tgt_all,
             R_spec_all=R_spec_all, R_diff_all=R_diff_all, f_tgt_all=f_tgt_all, f_diff_all=f_diff_all, n_d0=n_d0+1, theta=theta_out_BRDF, phi=phi_out_BSDF)