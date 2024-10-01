import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
from itertools import product
import time
import subprocess

thickness = np.linspace(100000, 1000000, 10)
f_vol_tot = np.linspace(0.001, 0.01, 19)

wvl = 11
n_theta = 181
n_phi = 360
wavelength = np.linspace(440, 460, wvl) # in nm
mat_particle = np.array(['TiO2_Sarkar','SiO2_bulk']*5) # particle only, from out to in

data_sweep = np.array(list(product(thickness, f_vol_tot)))
n_core = 384

identifier = 'theta160_phi0_lambda450'

with open(directory + '/monte_carlo_batch.sh', 'w') as batch_file:
    ### For MIT Supercloud
    batch_file.write("#!/bin/bash\n\n")
    batch_file.write("#SBATCH -n " + str(int(n_core)) + "\n\n")
    batch_file.write("source /etc/profile\n")
    batch_file.write("module load anaconda/2023a\n")
    batch_file.write("export OMP_NUM_THREADS=1\n\n")
    batch_file.write("mpirun -np " + str(int(n_core)) + " python " + directory + "/run_monte_carlo_parameter_sweep.py")

proc = subprocess.Popen(["chmod", "u+x", directory + '/monte_carlo_batch.sh']) 

if not os.path.isdir(directory[:-9] + '/results'):
    os.mkdir(directory[:-9] + '/results')
if not os.path.isdir(directory[:-9] + '/data'):
    os.mkdir(directory[:-9] + '/data')

if os.path.exists(directory + '/results/' + 'MC_param_sweep_' + identifier + '.npz'):
    prev_data = np.load(directory + '/results/' + 'MC_param_sweep_' + identifier + '.npz')
    BRDF_all = prev_data['BRDF_all']
    R_tgt_all = prev_data['R_tgt_all']
    R_all = prev_data['R_all']
    n_d0 = prev_data['n_d0']
else:
    BRDF_spec_all = np.zeros((n_theta, n_phi, data_sweep.shape[0]))
    BRDF_diff_all = np.zeros((n_theta, n_phi, data_sweep.shape[0]))
    R_tgt_all = np.zeros(data_sweep.shape[0])
    R_spec_all = np.zeros(data_sweep.shape[0])
    R_diff_all = np.zeros(data_sweep.shape[0])
    f_spec_all = np.zeros(data_sweep.shape[0])
    n_d0 = 0
    
for n_d in range(n_d0, data_sweep.shape[0]):
    thickness_sweep = data_sweep[n_d,0]
    f_vol_tot_sweep = np.array([data_sweep[n_d,1]])

    print('\n### Simulation ' + str(n_d), flush=True)
    print('    | Film Thickness: ' + str(thickness_sweep), flush=True)
    print('    | Total Particle Volume Fraction: ' + str(f_vol_tot_sweep[0]), flush=True)

    np.savez(directory + '/data/sweep_parameters', thickness=thickness_sweep, f_vol_tot=f_vol_tot_sweep, index=n_d,
             mat_particle=mat_particle, wavelength=wavelength)   

    while True:
        if not os.path.exists(directory[:-9] + '/data/dual_particle_sweep_BRDF' + str(n_d) + '.npz'):
            t1 = time.time()
            proc = subprocess.Popen(["sbatch", directory[:-9] + '/monte_carlo_batch.sh'])
            while True:
                if os.path.exists(directory[:-9] + '/data/dual_particle_sweep_BRDF' + str(n_d) + '.npz'):
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
            
        with np.load(directory[:-9] + '/data/dual_particle_sweep_BRDF' + str(n_d) + '.npz') as load_data:
            BRDF_spec = load_data['BRDF_spec']
            BRDF_diff = load_data['BRDF_diff']
            wavelength = load_data['wavelength']
        
        if np.sum(np.isnan(R_spec + R_diff + R_scat)) == 0:
            break
        else:
            os.remove(directory[:-9] + '/data/dual_particle_sweep_BRDF' + str(n_d) + '.npz')
    
    I_solar = rc.load_solar('AM1.5_SMARTS295', 1, wavelength)
    
    R_all[:,n_d] = R_spec + R_diff + R_scat
    A_all[:,n_d] = T_ball + T_diff + T_scat + A_medium + A_particle + A_TIR
    P_tot = 0
    for l in range(wavelength.size - 1):
        dlam = wavelength[l+1] - wavelength[l]
        P_ref_all[n_d] += 0.5*(R_all[l,n_d]*I_solar[l] + R_all[l+1,n_d]*I_solar[l+1])*dlam
        P_abs_all[n_d] += 0.5*(A_all[l,n_d]*I_solar[l] + A_all[l+1,n_d]*I_solar[l+1])*dlam
        if wavelength[l+1] <= 400 or wavelength[l] >= 700:
            P_ref_UVIR_all[n_d] += 0.5*(R_all[l,n_d]*I_solar[l] + R_all[l+1,n_d]*I_solar[l+1])*dlam
        else:
            P_ref_vis_all[n_d] += 0.5*(R_all[l,n_d]*I_solar[l] + R_all[l+1,n_d]*I_solar[l+1])*dlam
        P_tot += 0.5*(I_solar[l] + I_solar[l+1])*dlam
        
    print('\n    | Reflected Power: %.2f W/m^2, %.2f percent' %(P_ref_all[n_d], 100*P_ref_all[n_d]/P_tot), flush=True)
    print('    | Refected Power Ratio (uvir/vis): %.2f ' %(P_ref_UVIR_all[n_d]/P_ref_vis_all[n_d]), flush=True)
    print('    | Transmitted Power: %.2f W/m^2, %.2f percent' %(P_abs_all[n_d], 100*P_abs_all[n_d]/P_tot), flush=True)

    if not os.path.isdir(directory + '/results'):
        os.mkdir(directory + '/results')
    np.savez(directory + '/results/param_sweep', params=data_sweep, R_all=R_all, A_all=A_all, P_ref=P_ref_all, P_abs=P_abs_all, f_vol=f_vol_all, r_list=r_list_all,
             P_ref_vis=P_ref_vis_all, P_ref_UVIR=P_ref_UVIR_all, n_d0=n_d0+1)
