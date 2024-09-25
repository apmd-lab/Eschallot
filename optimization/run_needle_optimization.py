import numpy as np
import read_mat_data as rmd
import needle_algorithm as needle
import time
import os
directory = os.path.dirname(os.path.realpath(__file__))
import multiprocessing
import psutil

def run_needle(index, mat_dict, mat_needle, r, n, ban_needle, lam, theta, phi, pol, Q_sca_con, Q_abs_con, Q_ext_con, p_con,
               d_low=10):
    ml_init = needle.multilayer(r, n, ban_needle, lam, theta, phi, pol)
    ml_init.update(r, n)

    iteration = 1
    r_new, cost, Q_sca_new, Q_abs_new, Q_ext_new, p_new = needle.refine_r(index, ml_init, r, n, lam, Q_sca_con, Q_abs_con, Q_ext_con,
                                                                          p_con, d_low=d_low)
    n_new = n.copy()
    ban_needle_new = ban_needle.copy()
    needle_status = 1
    cost_count = 0
    cost_prev = np.inf
    while True:
        iteration += 1
        needle_status, n_needle, loc, dMF = needle.insert_needle(ml_init, mat_dict, mat_needle, r_new, n_new, ban_needle_new, lam,
                                                                 Q_sca_new, Q_abs_new, Q_ext_new, p_new,
                                                                 Q_sca_con, Q_abs_con, Q_ext_con, p_con, d_low)
        if needle_status == 0:
            break
        
        n_new, r_new, ban_needle_new, Q_sca_new, Q_abs_new, Q_ext_new, p_new = needle.deep_search(index, ml_init, mat_needle, n_needle,
                                                                                                  loc, dMF, r_new, n_new, ban_needle_new,
                                                                                                  Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                                                  d_low=d_low)
        
        if cost < 0.99*cost_prev:
            cost_prev = cost
        else:
            cost_count += 1
        if cost_count > 1:
            break
    iteration += 1
    
    # Clean up layers that are too thin
    thin_layer = 1
    while thin_layer:
        r_fin = r_new[0]
        n_fin = n_new[:,0].reshape(np.size(lam), 1)
        thin_layer = 0
        for l in range(r_new.size-1):
            if r_new[l]-r_new[l+1] > d_low:
                r_fin = np.append(r_fin, r_new[l+1])
                n_fin = np.concatenate((n_fin, n_new[:,l+1].reshape(np.size(lam), 1)), axis=1)
            else:
                thin_layer = 1
        if r_new[-1] > d_low:
            n_fin = np.concatenate((n_fin, n_new[:,-1].reshape(np.size(lam), 1)), axis=1)
        elif r_fin.size != 1:
            r_fin = r_fin[:-1]
            thin_layer = 1
        else:
            Q_sca_fin = Q_sca_new.copy()
            Q_abs_fin = Q_abs_new.copy()
            Q_ext_fin = Q_ext_new.copy()
            p_fin = p_new.copy()
            break
        
        if r_fin.size > 1:
            for l in range(r_fin.size - 1, -1, -1):
                if np.array_equal(n_fin[:,l+1], n_fin[:,l]):
                    n_fin = np.delete(n_fin, l+1, axis=1)
                    r_fin = np.delete(r_fin, l)
        r_new, cost, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin = needle.refine_r(index, ml_init, r_fin, n_fin, lam,
                                                                       Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                       d_low=d_low)
        n_new = n_fin.copy()
    
    r_fin = r_new.copy()
    n_fin = n_new.copy()
    
    return r_fin, n_fin, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, cost

def radius_sweep(r_min, r_max, N_sweep):
    # Clear Folder
    for filename in os.listdir(directory):
        if '.png' in filename:
            os.remove(directory + filename)
    
    # Variable setup code block
    mat_profile = np.array(['Air','TiO2_Sarkar']) # Top to bottom
    ban_needle = np.array([False]) # exclude medium
    mat_needle = np.array(['SiO2_bulk','TiO2_Sarkar']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik'
    lam = np.linspace(360, 830, 830-360+1)
    theta = np.linspace(0, 180, 37)*np.pi/180 # 0: forward, 180: backward
    phi = np.array([0,90])*np.pi/180
    polarization = 45*np.pi/180
    
    # Create n
    mat_type = list(set(np.hstack((mat_profile, mat_needle))))
    raw_wavelength, mat_dict = rmd.load_all(lam, 'n_k', mat_type)
    
    n = np.zeros((np.size(lam,0), np.size(mat_profile,0))).astype(complex)
    count = 0
    for mat in mat_profile:
        n[:,count] = mat_dict[mat]
        count += 1
    
    # 1st half columns: conditions for each angle of incidence
    # 2nd half columns: weights
    Q_sca_con = np.nan*np.ones((3, 2, lam.size)) # 0: equality, 1: lower than, 2: greater than
    Q_abs_con = np.nan*np.ones((3, 2, lam.size))
    Q_ext_con = np.nan*np.ones((3, 2, lam.size))
    p_con = np.nan*np.ones((3, 2, lam.size, theta.size, phi.size))
    
    # Scattering Efficiency
    Q_sca_con[2,0,290] = 0.1
    #------------------------------
    Q_sca_con[2,1,:] = 0.1
    
    # Absorption Efficiency
    # Q_abs_con[0,:190] = np.nan
    # Q_abs_con[0,190:191] = np.nan
    # Q_abs_con[0,191:] = np.nan
    #------------------------------
    # Q_abs_con[1,:] = 0.25
    
    # Extinction Efficiency
    # Q_ext_con[0,:] = np.nan
    
    # Phase Function
    p_con[0,0,290,:30,0] = 0
    p_con[0,0,290,32,0] = 5
    p_con[0,0,290,35:,0] = 0
    p_con[0,0,290,:,1] = 0
    #------------------------------
    p_con[0,1,:,:,:] = 1

    iterable = []
    cost = np.zeros(N_sweep)
    radius_list = np.linspace(r_min, r_max, N_sweep) # in nm
    for i in range(N_sweep):
        r_profile = np.array([radius_list[i]])
        temp_list = [i, mat_dict, mat_needle, r_profile, n, ban_needle,
                      lam, theta, phi, polarization,
                      Q_sca_con, Q_abs_con, Q_ext_con, p_con, 1.5]
        iterable.append(tuple(temp_list))

    pool = multiprocessing.Pool(processes=psutil.cpu_count()-1)
    Job = pool.starmap_async(run_needle, iterable)
    pool_result = Job.get()
    pool.close()
    pool.join()
    
    radius = dict()
    RI = dict()
    Q_sca = np.zeros((N_sweep, lam.size))
    Q_abs = np.zeros((N_sweep, lam.size))
    Q_ext = np.zeros((N_sweep, lam.size))
    p = np.zeros((N_sweep, lam.size, theta.size, phi.size))
    N_layer = np.zeros(N_sweep)
    for i in range(N_sweep):
        radius[i] = pool_result[i][0]
        N_layer[i] = pool_result[i][0].size
        RI[i] = pool_result[i][1]
        Q_sca[i,:] = pool_result[i][2]
        Q_abs[i,:] = pool_result[i][3]
        Q_ext[i,:] = pool_result[i][4]
        p[i,:,:,:] = pool_result[i][5]
        cost[i] = pool_result[i][6]
    
    r_save = np.zeros((N_sweep, int(np.max(N_layer))))
    n_save = np.zeros((N_sweep, lam.size, int(np.max(N_layer))+1)).astype(np.complex128)
    for i in range(N_sweep):
        r_save[i,:int(N_layer[i])] = radius[i]
        n_save[i,:,:int(N_layer[i])+1] = RI[i]
    
    cost_sort = np.argsort(cost)
    r_save = r_save[cost_sort,:]
    n_save = n_save[cost_sort,:,:]
    Q_sca = Q_sca[cost_sort,:]
    Q_abs = Q_abs[cost_sort,:]
    Q_ext = Q_ext[cost_sort,:]
    p = p[cost_sort,:,:,:]
    N_layer = N_layer[cost_sort]
    
    np.savez(directory + '\\result_data', r=r_save, n=n_save,
             Q_sca=Q_sca, Q_abs=Q_abs, Q_ext=Q_ext, p=p, N_layer=N_layer)

if __name__ == '__main__':
    T1 = time.time()
    radius_sweep(r_min=700, r_max=750, N_sweep=26)
    T2 = time.time()
    print('Total time: ' + str(T2 - T1))