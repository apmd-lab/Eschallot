import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import subprocess
import numpy as np
import radiative_cooling.cooling_performance as rc
import time
import psutil

def FoM(index, index_disc, pht_per_wvl=1e3, lam1=300, lam2=2500, lam_step=50, n_core=168):
    filename = directory + '/data/CP_PSO_BRDF.npz'
    if os.path.exists(filename):
        os.remove(filename)
    
    layer_thickness = index[0]
    f_vol_tot = index[1]
    f_num1 = index[2]
    r_mean1 = index[3]
    r_mean2 = index[4]
    scale1 = index[5]
    scale2 = index[6]
    
    mat1 = np.argwhere(index_disc[0,:]==1)[0][0]
    mat2 = np.argwhere(index_disc[1,:]==1)[0][0]

    np.savez(directory + '/data/CP_PSO_optimization_parameters', layer_thickness=layer_thickness, f_vol_tot=f_vol_tot, f_num1=f_num1, r_mean1=r_mean1, r_mean2=r_mean2,
             scale1=scale1, scale2=scale2, mat1=mat1, mat2=mat2, pht_per_wvl=pht_per_wvl, lam1=lam1, lam2=lam2, lam_step=lam_step)   

    ### CHANGE THIS TO BE SUITABLE FOR YOUR SERVER CONFIGURATION ###
    with open(directory + '/CP_PSO_monte_carlo_batch.sh', 'w') as batch_file:
        batch_file.write("#!/bin/bash\n\n")
        batch_file.write("#SBATCH -n " + str(int(n_core)) + "\n\n")
        batch_file.write("export OMP_NUM_THREADS=1\n\n")
        batch_file.write("mpirun -np " + str(int(n_core)) + " python " + directory + "/run_monte_carlo_CP_PSO.py")
    ################################################################
    
    while True:
        t1 = time.time()
        proc = subprocess.Popen(["sbatch", directory + '/CP_PSO_monte_carlo_batch.sh']) 
        while True:
            if os.path.exists(filename):
                break
            time.sleep(10)
        t2 = time.time()
        print('### Simulation Time: ' + str(t2 - t1) + ' s', flush=True)
    
        queue = subprocess.Popen(["squeue"], stdout=subprocess.PIPE)
        queue_str = queue.communicate()
        jobid_temp = queue_str[0].decode('utf-8').split()
        jobid = None
        
        for i in range(len(jobid_temp)):
            ### CHANGE THE BELOW CONDITIONAL TO REFLECT YOUR USERNAME ###
            if jobid_temp[i] == 'CP_PSO_m' and jobid_temp[i+1] == 'smin': 
                jobid = jobid_temp[i-2]
                break
        
        if jobid is not None:
            quit_proc = subprocess.Popen(["scancel", jobid])
            
        with np.load(filename) as load_data:
            BRDF_spec = load_data['BRDF_spec'] 
            BRDF_diff = load_data['BRDF_diff']
            wavelength = load_data['wavelength']
        
        BRDF = BRDF_spec + BRDF_diff
        if np.sum(np.isnan(BRDF)) == 0:
            break
        else:
            os.remove(filename)
    
    R = np.sum(BRDF[:,0,:-1,:], axis=(1, 2))
    lam = np.hstack((wavelength, np.array([2501,7999,8000,13000,13001,20000])))
    R = np.hstack((R, np.array([1,1,0,0,1,1])))
    cooler = rc.radiative_cooler(R, lam, season='summer', diffuse_R_sol=1)
    P_net, P_rad, P_atm, P_cond, P_sol1 = cooler.cooling_P(300, 300, hc=6)
    
    lam = np.array([300,4000,4001,7999,8000,13000,13001,20000])
    R = np.array([0,0,1,1,0,0,1,1])
    cooler = rc.radiative_cooler(R, lam, season='summer', diffuse_R_sol=1)
    P_net, P_rad, P_atm, P_cond, P_sol2 = cooler.cooling_P(300, 300, hc=6)
        
    print('\n    | Absorbed Power: %.2f W/m^2, %.2f percent' %(P_sol1, P_sol1/P_sol2), flush=True)
    
    FoM = -P_sol1

    if not os.path.isdir(directory + '/data'):
        os.mkdir(directory + '/data')
    np.savez(directory + '/data/fom_CP_PSO', FoM=FoM, BRDF=BRDF, r_core=r_core, d_shell=d_shell, f_vol=f_vol, mat_core=mat_core, mat_shell=mat_shell, P_sol1=P_sol1, P_sol2=P_sol2)
            
    return FoM
