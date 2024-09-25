import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
from itertools import product
import radiative_cooling.cooling_performance as rc
import time
import subprocess

material_name = np.array(['TiO2_Siefke','BaSO4_li','ZrO2_li'])
material1 = np.arange(material_name.size)
r_mean1 = np.array([50,150,250])
scale1 = 10**np.linspace(-1, 1, 2)
material2 = np.arange(material_name.size)
r_mean2 = np.array([50,150,250])
scale2 = 10**np.linspace(-1, 1, 2)
thickness = np.array([100000,200000,400000])
f_vol_tot = np.linspace(0.1, 0.5, 5)
f_num1 = np.linspace(0.5, 0.75, 1)

data_sweep = np.array(list(product(material1, r_mean1, scale1, material2, r_mean2, scale2, thickness, f_vol_tot, f_num1)))
n_core = 96

with open(directory + '/monte_carlo_batch.sh', 'w') as batch_file:
    batch_file.write("#!/bin/bash\n\n")
    batch_file.write("#SBATCH -n " + str(int(n_core)) + "\n\n")
    batch_file.write("source /etc/profile\n")
    batch_file.write("module load anaconda/2023a\n")
    batch_file.write("export OMP_NUM_THREADS=1\n\n")
    batch_file.write("mpirun -np " + str(int(n_core)) + " python " + directory + "/run_monte_carlo_parameter_sweep.py")

proc = subprocess.Popen(["chmod", "u+x", directory + '/monte_carlo_batch.sh']) 

if os.path.exists(directory + '/results/param_sweep.npz'):
    prev_data = np.load(directory + '/results/param_sweep.npz')
    R_all = prev_data['R_all']
    A_all = prev_data['A_all']
    P_ref_all = prev_data['P_ref']
    P_abs_all = prev_data['P_abs']
    P_ref_vis_all = prev_data['P_ref_vis']
    P_ref_UVIR_all = prev_data['P_ref_UVIR']
    f_vol_all = prev_data['f_vol']
    r_list_all = prev_data['r_list']
else:
    R_all = np.zeros((45, data_sweep.shape[0]))
    A_all = np.zeros((45, data_sweep.shape[0]))
    P_ref_all = np.zeros(data_sweep.shape[0])
    P_abs_all = np.zeros(data_sweep.shape[0])
    P_ref_vis_all = np.zeros(data_sweep.shape[0])
    P_ref_UVIR_all = np.zeros(data_sweep.shape[0])
    f_vol_all = np.zeros((2, 50, data_sweep.shape[0]))
    r_list_all = np.zeros((50, data_sweep.shape[0]))
    
for n_d in range(4088, data_sweep.shape[0]):
    materials_sweep = material_name[data_sweep[n_d,[0,3]].astype(int)]
    r_mean_sweep = data_sweep[n_d,[1,4]].reshape(1,-1)
    scale_sweep = data_sweep[n_d,[2,5]].reshape(1,-1)
    thickness_sweep = data_sweep[n_d,6]
    f_vol_tot_sweep = np.array([data_sweep[n_d,7]])
    f_num_sweep = np.array([data_sweep[n_d,8],1-data_sweep[n_d,8]]).reshape(1,-1)

    print('\n### Simulation ' + str(n_d), flush=True)
    for i in range(2):
        print('    | Material ' + str(i) + ': ' + materials_sweep[i], flush=True)
        print('    | Mean Radius ' + str(i) + ': ' + str(r_mean_sweep[0,i]), flush=True)
        print('    | Scale ' + str(i) + ': ' + str(scale_sweep[0,i]), flush=True)
    print('    | Film Thickness: ' + str(thickness_sweep), flush=True)
    print('    | Total Particle Volume Fraction: ' + str(f_vol_tot_sweep[0]), flush=True)
    print('    | Particle 1 Number Fraction: ' + str(f_num_sweep[0,0]), flush=True)

    if not os.path.isdir(directory + '/data'):
        os.mkdir(directory + '/data')
    np.savez(directory + '/data/sweep_parameters', materials=materials_sweep, r_mean=r_mean_sweep, scale=scale_sweep, thickness=thickness_sweep,
             f_vol_tot=f_vol_tot_sweep, f_num=f_num_sweep, index=n_d)   

    while True:
        if not os.path.exists(directory + '/data/param_sweep/dual_particle_sweep_BRDF' + str(n_d) + '.npz'):
            t1 = time.time()
            proc = subprocess.Popen(["sbatch", directory + '/monte_carlo_batch.sh'])
            while True:
                if os.path.exists(directory + '/data/param_sweep/dual_particle_sweep_BRDF' + str(n_d) + '.npz'):
                    break
                time.sleep(10)
            t2 = time.time()
            print('    | Simulation Time: ' + str(t2 - t1) + ' s', flush=True)
        
            queue = subprocess.Popen(["squeue"], stdout=subprocess.PIPE)
            queue_str = queue.communicate()
            jobid_temp = queue_str[0].decode('utf-8').split()
            jobid = None
            
            for i in range(len(jobid_temp)):
                if jobid_temp[i] == 'monte_ca' and jobid_temp[i+1] == 'smin':
                    jobid = jobid_temp[i-2]
                    break
            
            if jobid is not None:
                quit_proc = subprocess.Popen(["scancel", jobid])
            
        with np.load(directory + '/data/param_sweep/dual_particle_sweep_BRDF' + str(n_d) + '.npz') as load_data:
            R_spec = load_data['R_spec'] 
            R_diff = load_data['R_diff']
            R_scat = load_data['R_scat']
            T_ball = load_data['T_ball']
            T_diff = load_data['T_diff']
            T_scat = load_data['T_scat']
            A_medium = load_data['A_medium']
            A_particle = load_data['A_particle']
            A_TIR = load_data['A_TIR']
            wavelength = load_data['wavelength']
            f_vol = load_data['f_vol']
            r_list = load_data['r_list']
        
        if np.sum(np.isnan(R_spec + R_diff + R_scat + T_ball + T_diff + T_scat + A_medium + A_particle + A_TIR)) == 0:
            break
        else:
            os.remove(directory + '/data/param_sweep/dual_particle_sweep_BRDF' + str(n_d) + '.npz')
    
    f_vol_all[:,:,n_d] = f_vol[0,:,:]
    r_list_all[:,n_d] = r_list
    
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
             P_ref_vis=P_ref_vis_all, P_ref_UVIR=P_ref_UVIR_all)
