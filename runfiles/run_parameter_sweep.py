import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
from scipy.optimize import fsolve
from itertools import product
import time
import subprocess

import mie.simulate_particle as sim
import color.color_coordinates as cie

thickness = np.array([100000])
# Note: thickness at which x% of photons scatter at least once (the number is the extinction cross section) -> -np.log(1-x)*137891.8083528611
f_vol_tot = np.array([0.05])
init_theta = np.array([30,60,85])*np.pi/180

wvl = 1
n_theta = 121 # 121 / 61
n_phi = 18 # 18 / 12
#wavelength = np.array([[380,390,400,410,420,430,440,445,447.5,450],
#                       [452.5,455,460,470,480,490,500,510,520,530],
#                       [540,550,560,570,580,590,600,610,620,630],
#                       [640,650,660,670,680,690,700,710,720,730]]) # in nm / np.array([400,440,447,450,453,500,600,700])
wavelength = np.array([[450]])
init_phi = np.array([0])*np.pi/180
nu = np.linspace(0, 1, n_theta) # for even area spacing along theta (number of angles must be odd to always include pi/2)
phi_in_BSDF = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
phi_out_BSDF = np.linspace(0, 2*np.pi, n_phi, endpoint=False) # set azimuthal angle sampling points
pht_per_wvl = 5e6 # number of photons (on average) to simulate for each sampled wavelength (3e6~7 for 1mm)

data_sweep = np.array(list(product(thickness, f_vol_tot, init_theta, np.arange(wavelength.shape[0]))))
pht_ratio = np.array(list(product(thickness/np.min(thickness), f_vol_tot/np.min(f_vol_tot), init_theta, np.arange(wavelength.shape[0]))))
pht_per_wvl = pht_per_wvl/(pht_ratio[:,0]*pht_ratio[:,1])
n_core = 288

identifier = np.array(['theta180_phi_all_lambda450_rmax6000_dtheta5_Bwd'])
suffix = '_retroreflector'

incidence = 'retro' # normal / oblique / multi / color / retro

# Import Optimized Particle Data
theta_pf = np.linspace(0, np.pi, 361)
phi_pf = np.array([0,90])*np.pi/180
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

# Determine Optimal Number Fractions -------------------------------------------------------
#def equations(x, f1, f2, f3, C1, C2, C3):
#    N1, N2, N3 = x
#    C11, C12, C13 = C1
#    C21, C22, C23 = C2
#    C31, C32, C33 = C3
#
#    eq1 = N1 + N2 + N3 - 1
#    eq2 = (f1 * C11 * N1) / (C11 * N1 + C21 * N2 + C31 * N3) - (f2 * C22 * N2) / (C12 * N1 + C22 * N2 + C32 * N3)
#    eq3 = (f1 * C11 * N1) / (C11 * N1 + C21 * N2 + C31 * N3) - (f3 * C33 * N3) / (C13 * N1 + C23 * N2 + C33 * N3)
#
#    print('Residuals:')
#    print(eq1)
#    print(eq2)
#    print(eq3)
#    return [eq1, eq2, eq3]
#
## Example parameters
#f1 = np.sum(pf_B[0,291:302,0]*np.sin(theta_pf[291:302]))/np.sum(pf_B[0,:,:]*np.sin(theta_pf)[:,np.newaxis])
#f2 = np.sum(pf_G[1,291:302,0]*np.sin(theta_pf[291:302]))/np.sum(pf_G[1,:,:]*np.sin(theta_pf)[:,np.newaxis])
#f3 = np.sum(pf_R[2,291:302,0]*np.sin(theta_pf[291:302]))/np.sum(pf_R[2,:,:]*np.sin(theta_pf)[:,np.newaxis])
#
## Initial guess for N1, N2, N3
#initial_guess = [1/3, 1/3, 1/3]
#
## Solve the system of equations
#print(f1)
#print(f2)
#print(f3)
#print(C1)
#print(C2)
#print(C3)
#solution = fsolve(equations, initial_guess, args=(f1, f2, f3, C1, C2, C3))
#
## Display the solution
#N1, N2, N3 = solution
#f_num = np.array([N1,N2,N3])
#print('### Optimal Particle Number Fraction: ' + str(N1) + ' / ' + str(N2) + ' / ' + str(N3))
# -------------------------------------------------------------------------------------------

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
    XYZ_all = prev_data['XYZ_all']
    n_d0 = prev_data['n_d0']
else:
    BRDF_spec_all = np.zeros((data_sweep.shape[0], wvl, int((n_theta-1)/2+2), n_phi))
    BRDF_diff_all = np.zeros((data_sweep.shape[0], wvl, int((n_theta-1)/2+2), n_phi))
    R_tgt_all = np.zeros((data_sweep.shape[0], wvl))
    R_spec_all = np.zeros((data_sweep.shape[0], wvl))
    R_diff_all = np.zeros((data_sweep.shape[0], wvl))
    f_tgt_all = np.zeros((data_sweep.shape[0], wvl))
    f_diff_all = np.zeros((data_sweep.shape[0], wvl))
    XYZ_all = np.zeros((data_sweep.shape[0], int((n_theta-1)/2), n_phi, 3))
    sRGB_all = np.zeros((data_sweep.shape[0], int((n_theta-1)/2), n_phi, 3))
    n_d0 = 0

# Parameter Sweep
for n_d in range(n_d0, data_sweep.shape[0]):
    thickness_sweep = data_sweep[n_d,0]
    f_vol_tot_sweep = np.array([data_sweep[n_d,1]])
    init_theta_sweep = np.array([data_sweep[n_d,2]])
    wavelength_sweep = wavelength[int(data_sweep[n_d,3]),:]
    pht_per_wvl_sweep = pht_per_wvl[n_d]

    print('\n### Simulation ' + str(n_d), flush=True)
    print('    | Film Thickness: ' + str(thickness_sweep), flush=True)
    print('    | Total Particle Volume Fraction: ' + str(f_vol_tot_sweep[0]), flush=True)
    print('    | Angle of Incidence (polar): ' + str(init_theta_sweep[0]*180/np.pi), flush=True)
    print('    | Wavelength Range: ' + str(wavelength_sweep[0]) + ' - ' + str(wavelength_sweep[-1]), flush=True)
    print('    | Photons Per Wavelength: ' + str(pht_per_wvl_sweep), flush=True)

    np.savez(directory[:-9] + '/data/sweep_parameters',
             thickness=thickness_sweep,
             f_vol_tot=f_vol_tot_sweep,
             mat_particle=mat_particle,
             r_particle=r_particle,
             #f_num=f_num,
             index=n_d,
             wavelength=wavelength_sweep,
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
            theta_out_BRDF_edge = load_data['theta_out_BRDF_edge']
        
        if np.sum(np.isnan(BRDF_spec + BRDF_diff)) == 0:
            break
        else:
            os.remove(directory[:-9] + '/data/param_sweep_' + identifier[0]  + suffix + '_BSDF' + str(n_d) + '.npz')
    
    BRDF_spec_all[n_d,:,:,:] = BRDF_spec[:,0,0,:,:]
    BRDF_diff_all[n_d,:,:,:] = BRDF_diff[:,0,0,:,:]
    
    R_lambertian = np.zeros((int((n_theta-1)/2), n_phi))
    for nth in range(int((n_theta-1)/2)):
        for nph in range(n_phi):
            if nph != n_phi - 1:
                R_lambertian[nth,nph] = -(phi_out_BSDF[nph+1] - phi_out_BSDF[nph])/(4*np.pi)*(np.cos(2*theta_out_BRDF_edge[nth]) - np.cos(2*theta_out_BRDF_edge[nth+1]))
            else:
                R_lambertian[nth,nph] = -(phi_out_BSDF[0] + 2*np.pi - phi_out_BSDF[nph])/(4*np.pi)*(np.cos(2*theta_out_BRDF_edge[nth]) - np.cos(2*theta_out_BRDF_edge[nth+1]))
    
    if incidence == 'normal':
        mask_theta_tgt = (theta_out_BRDF >= 140*np.pi/180)*(theta_out_BRDF <= 160*np.pi/180)
        mask_phi_tgt = (phi_out_BSDF >= 45*np.pi/180)*(phi_out_BSDF <= 135*np.pi/180) + (phi_out_BSDF >= 225*np.pi/180)*(phi_out_BSDF <= 315*np.pi/180)
    elif incidence == 'oblique':
        mask_theta_tgt = (theta_out_BRDF <= (180 - 13.3)*np.pi/180)*(theta_out_BRDF >= (180 - 33.3)*np.pi/180)
        mask_theta_tgt[-1] = False
        mask_phi_tgt = (phi_out_BSDF >= 135*np.pi/180)*(phi_out_BSDF <= 225*np.pi/180)
    elif incidence == 'multi':
        mask_theta_tgt = theta_out_BRDF >= (180 - 17)*np.pi/180
        mask_theta_tgt[-1] = False
        mask_phi_tgt = phi_out_BSDF > 0
    elif incidence == 'color':
        mask_theta_tgt = theta_out_BRDF >= (180 - 10)*np.pi/180
        mask_theta_tgt[-1] = False
        mask_phi_tgt = phi_out_BSDF > 0
        
        wavelength_cie = np.linspace(380, 730, 351)
        BRDF_cie = (BRDF_spec_all + BRDF_diff_all).reshape(wavelength.size, int((n_theta-1)/2+2), n_phi)
        for nth in range(int((n_theta-1)/2)):
            for nph in range(n_phi):
                R_norm = BRDF_cie[:,nth,nph]/R_lambertian[nth,nph]
                R_norm = np.interp(wavelength_cie, wavelength.reshape(-1), R_norm)
                XYZ_all[n_d,nth,nph,:] = cie.color(R_norm, wavelength_cie, 'Global_Vert_AM1_5', column=1).CIE_XYZ()
                sRGB_all[n_d,nth,nph,:], _ = cie.color(R_norm, wavelength_cie, 'Global_Vert_AM1_5', column=1).sRGB()
    elif incidence == 'retro':
        mask_theta_tgt = (theta_out_BRDF >= np.pi - init_theta_sweep - 5*np.pi/180)*(theta_out_BRDF < np.pi - init_theta_sweep + 5*np.pi/180)
        mask_phi_tgt = (phi_out_BSDF <= 5*np.pi/180) + (phi_out_BSDF >= 355*np.pi/180)
        
    mask_tgt = (mask_theta_tgt[:,np.newaxis]*mask_phi_tgt[np.newaxis,:])[np.newaxis,:,:]
    R_tgt_all[n_d,:] = np.sum(BRDF_spec[:,0,0,:,:]*mask_tgt + BRDF_diff[:,0,0,:,:]*mask_tgt, axis=(1,2))
    R_spec_all[n_d,:] = np.sum(BRDF_spec[:,0,0,:-1,:], axis=(1,2))
    R_diff_all[n_d,:] = np.sum(BRDF_diff[:,0,0,:-1,:], axis=(1,2))
    f_tgt_all[n_d,:] = R_tgt_all[n_d,:]/(R_spec_all[n_d,:] + R_diff_all[n_d,:])
    f_diff_all[n_d,:] = R_tgt_all[n_d,:]/R_diff_all[n_d,:]
    
    print('\n    | Directional Reflectance: %.2f percent' %(R_tgt_all[n_d,0]), flush=True)
    print('    | Directional Reflectance Relative to All Reflectance: %.2f percent' %(f_tgt_all[n_d,0]), flush=True)
    print('    | Directional Reflectance Relative to Diffuse Reflectance: %.2f percent' %(f_diff_all[n_d,0]), flush=True)

    if incidence == 'color':
        np.savez(directory[:-9] + '/results/param_sweep_' + identifier[0] + suffix, params=data_sweep, BRDF_spec_all=BRDF_spec_all, BRDF_diff_all=BRDF_diff_all, mask_tgt=mask_tgt, R_tgt_all=R_tgt_all,
                 R_spec_all=R_spec_all, R_diff_all=R_diff_all, f_tgt_all=f_tgt_all, f_diff_all=f_diff_all, n_d0=n_d0+1, theta=theta_out_BRDF, phi=phi_out_BSDF, theta_edge=theta_out_BRDF_edge,
                 XYZ_all=XYZ_all, sRGB_all=sRGB_all, BRDF_cie=BRDF_cie, R_lambertian=R_lambertian)
    else:
        np.savez(directory[:-9] + '/results/param_sweep_' + identifier[0] + suffix, params=data_sweep, BRDF_spec_all=BRDF_spec_all, BRDF_diff_all=BRDF_diff_all, mask_tgt=mask_tgt, R_tgt_all=R_tgt_all,
                 R_spec_all=R_spec_all, R_diff_all=R_diff_all, f_tgt_all=f_tgt_all, f_diff_all=f_diff_all, n_d0=n_d0+1, theta=theta_out_BRDF, phi=phi_out_BSDF, theta_edge=theta_out_BRDF_edge,
                 XYZ_all=XYZ_all, sRGB_all=sRGB_all, R_lambertian=R_lambertian)