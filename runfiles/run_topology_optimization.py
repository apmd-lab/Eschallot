import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-9])

import numpy as np
import util.read_mat_data as rmd
import optimization.topology_optimization as topopt
import time
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
import psutil

import warnings
warnings.filterwarnings('ignore')

def radius_sweep(identifier, r_min, r_max, N_sweep, d_low, max_layers):
    # Clear Folder
    for filename in os.listdir(directory):
        if '.png' in filename:
            os.remove(directory + filename)
    
    # Variable setup code block
    mat_profile = np.array(['PMMA','TiO2_Sarkar']) # Outer(including embedding medium) to inner
    mat_needle = np.array(['SiO2_bulk','TiO2_Sarkar']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik'
    lam_cost = np.array([450])
    theta_cost = np.linspace(0, 180, 91)*np.pi/180 # 0: forward, 180: backward
    phi_cost = np.array([0,90])*np.pi/180
    lam_plot = np.linspace(440, 460, 11)
    theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
    phi_plot = np.array([0,90])*np.pi/180
    polarization = 45*np.pi/180
    
    # Create n
    mat_type = list(set(np.hstack((mat_profile, mat_needle))))
    raw_wavelength, mat_dict_cost = rmd.load_all(lam_cost, 'n_k', mat_type)
    raw_wavelength, mat_dict_plot = rmd.load_all(lam_plot, 'n_k', mat_type)
    
    n = np.zeros((np.size(lam_cost,0), np.size(mat_profile,0))).astype(complex)
    count = 0
    for mat in mat_profile:
        n[:,count] = mat_dict_cost[mat]
        count += 1
    
    # 1st index: 0: equality, 1: lower than, 2: greater than, 3: weights only (non-least-squares/no target value)
    # 2nd index: conditions for each angle of incidence
    Q_sca_con = np.zeros((4, 2, lam_cost.size))
    Q_abs_con = np.zeros((4, 2, lam_cost.size))
    Q_ext_con = np.zeros((4, 2, lam_cost.size))
    p_con = np.zeros((4, 2, lam_cost.size, theta_cost.size, phi_cost.size))
    diff_CS_con = np.zeros((4, 2, lam_cost.size, theta_cost.size, phi_cost.size))
    
    # Scattering Efficiency
    Q_sca_con[2,0,0] = 0.1
    #------------------------------
    Q_sca_con[2,1,0] = 10
    
    # Absorption Efficiency
    # Q_abs_con[0,:190] = 0
    # Q_abs_con[0,190:191] = 0
    # Q_abs_con[0,191:] = 0
    #------------------------------
    # Q_abs_con[1,:] = 0.25
    
    # Extinction Efficiency
    # Q_ext_con[0,:] = 0
    
    # Phase Function
    p_con[0,0,0,1:32,0] = 0 # 0deg
    p_con[0,0,0,32,0] = 10 # 160deg
    p_con[0,0,0,33:,0] = 0
    p_con[0,0,0,1:,1] = 0
    #------------------------------
    p_con[0,1,0,1:,:] = 1
    
    # Differential Scattering Cross Section
    # diff_CS_con[3,1,:,:,:] = 0

    ### Initial Single-Layer Optimization to Eliminate Redundant Runs
    # Distribute Radii for Sweeping
    radius_list = np.linspace(r_min, r_max, N_sweep) # in nm
    
    quo, rem = divmod(N_sweep, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size+1)])
    
    radius_list_proc = radius_list[data_disp[comm.rank]:data_disp[comm.rank+1]]

    # Run Optimization
    if comm.rank == 0:
        print('### Initial Single-Layer Optimization (N_sweep = ' + str(N_sweep) + ')', flush=True)
        print('    Progress: ', end='', flush=True)
    
    ban_needle = np.array([True]) # Outer(excluding embedding medium) to inner
    radius_init_proc = np.zeros(data_size[comm.rank])
    for nr in range(data_size[comm.rank]):
        r_init, _, _, _, _, _, _, _ = topopt.run_needle(comm.rank,
                                                        mat_dict_cost,
                                                        mat_dict_plot,
                                                        mat_needle,
                                                        mat_profile,
                                                        np.array([radius_list_proc[nr]]),
                                                        n,
                                                        ban_needle,
                                                        lam_cost,
                                                        theta_cost,
                                                        phi_cost,
                                                        lam_plot,
                                                        theta_plot,
                                                        phi_plot,
                                                        polarization,
                                                        Q_sca_con,
                                                        Q_abs_con,
                                                        Q_ext_con,
                                                        p_con,
                                                        diff_CS_con,
                                                        d_low,
                                                        r_max,
                                                        max_layers)
        
        assert r_init.size == 1
        
        radius_init_proc[nr] = r_init[0]

    print('/', end='', flush=True)

    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])

    radius_init = np.zeros(N_sweep)
    comm.Allgatherv(radius_init_proc, [radius_init, data_size, data_disp, MPI.DOUBLE])
    
    radius_list = radius_init[0].reshape(1)
    for ns in range(N_sweep-1):
        if ns == 0:
            if np.abs(radius_init[ns+1] - radius_list) > 1e-3*radius_list:
                radius_list = np.append(radius_list, radius_init[ns+1])
        else:
            redundant = np.any(np.abs(radius_init[ns+1] - radius_list) <= 1e-3*radius_init[ns+1])
            if not redundant:
                radius_list = np.append(radius_list, radius_init[ns+1])
    N_sweep = radius_list.size

    #np.savez(directory[:-9] + "/debug/radius_list", radius_list=radius_list)
    
    ### Optimization of Non-Redundant Radii
    # Distribute Radii for Sweeping
    quo, rem = divmod(N_sweep, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size+1)])
    
    radius_list_proc = radius_list[data_disp[comm.rank]:data_disp[comm.rank+1]]

    # Run Optimization
    if comm.rank == 0:
        print('\n### Topology Optimization (N_sweep = ' + str(N_sweep) + ')', flush=True)
        print('    Progress: ', end='', flush=True)
    
    ban_needle = np.array([False]) # Outer(excluding embedding medium) to inner
    radius_proc = dict()
    RI_proc = dict()
    Q_sca_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    Q_abs_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    Q_ext_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    p_proc = np.zeros((data_size[comm.rank], lam_plot.size, theta_plot.size, phi_plot.size))
    diff_CS_proc = np.zeros((data_size[comm.rank], lam_plot.size, theta_plot.size, phi_plot.size))
    N_layer_proc = np.zeros(data_size[comm.rank])
    cost_proc = np.zeros(data_size[comm.rank])
    for nr in range(data_size[comm.rank]):
        r_fin, n_fin, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, diff_CS_fin, cost_fin = topopt.run_needle(comm.rank,
                                                                                                        mat_dict_cost,
                                                                                                        mat_dict_plot,
                                                                                                        mat_needle,
                                                                                                        mat_profile,
                                                                                                        np.array([radius_list_proc[nr]]),
                                                                                                        n,
                                                                                                        ban_needle,
                                                                                                        lam_cost,
                                                                                                        theta_cost,
                                                                                                        phi_cost,
                                                                                                        lam_plot,
                                                                                                        theta_plot,
                                                                                                        phi_plot,
                                                                                                        polarization,
                                                                                                        Q_sca_con,
                                                                                                        Q_abs_con,
                                                                                                        Q_ext_con,
                                                                                                        p_con,
                                                                                                        diff_CS_con,
                                                                                                        d_low,
                                                                                                        r_max,
                                                                                                        max_layers)
        
        radius_proc[nr] = r_fin
        N_layer_proc[nr] = r_fin.size
        RI_proc[nr] = n_fin
        Q_sca_proc[nr,:] = Q_sca_fin
        Q_abs_proc[nr,:] = Q_abs_fin
        Q_ext_proc[nr,:] = Q_ext_fin
        p_proc[nr,:,:,:] = p_fin
        diff_CS_proc[nr,:,:,:] = diff_CS_fin
        cost_proc[nr] = cost_fin

    print('/', end='', flush=True)

    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])

    N_layer = np.zeros(N_sweep)
    cost = np.zeros(N_sweep)
    comm.Allgatherv(N_layer_proc, [N_layer, data_size, data_disp, MPI.DOUBLE])
    comm.Gatherv(cost_proc, [cost, data_size, data_disp, MPI.DOUBLE], root=0)
        
    data_size_temp = data_size*lam_plot.size
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    Q_sca_temp = np.zeros(N_sweep*lam_plot.size)
    Q_abs_temp = np.zeros(N_sweep*lam_plot.size)
    Q_ext_temp = np.zeros(N_sweep*lam_plot.size)
    comm.Gatherv(Q_sca_proc.reshape(-1), [Q_sca_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(Q_abs_proc.reshape(-1), [Q_abs_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(Q_ext_proc.reshape(-1), [Q_ext_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    Q_sca = Q_sca_temp.reshape(N_sweep, lam_plot.size)
    Q_abs = Q_abs_temp.reshape(N_sweep, lam_plot.size)
    Q_ext = Q_ext_temp.reshape(N_sweep, lam_plot.size)
    
    data_size_temp = data_size*lam_plot.size*theta_plot.size*phi_plot.size
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    p_temp = np.zeros(N_sweep*lam_plot.size*theta_plot.size*phi_plot.size)
    diff_CS_temp = np.zeros(N_sweep*lam_plot.size*theta_plot.size*phi_plot.size)
    comm.Gatherv(p_proc.reshape(-1), [p_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(diff_CS_proc.reshape(-1), [diff_CS_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    p = p_temp.reshape(N_sweep, lam_plot.size, theta_plot.size, phi_plot.size)
    diff_CS = diff_CS_temp.reshape(N_sweep, lam_plot.size, theta_plot.size, phi_plot.size)

#    iterable = []
#    cost = np.zeros(N_sweep)
#    
#    for i in range(N_sweep):
#        r_profile = np.array([radius_list[i]])
#        temp_list = [i, mat_dict, mat_needle, r_profile, n, ban_needle,
#                      lam, theta, phi, polarization,
#                      Q_sca_con, Q_abs_con, Q_ext_con, p_con, 1.5]
#        iterable.append(tuple(temp_list))
#
#    pool = multiprocessing.Pool(processes=psutil.cpu_count()-1)
#    Job = pool.starmap_async(run_needle, iterable)
#    pool_result = Job.get()
#    pool.close()
#    pool.join()
#    
#    radius = dict()
#    RI = dict()
#    Q_sca = np.zeros((N_sweep, lam.size))
#    Q_abs = np.zeros((N_sweep, lam.size))
#    Q_ext = np.zeros((N_sweep, lam.size))
#    p = np.zeros((N_sweep, lam.size, theta.size, phi.size))
#    N_layer = np.zeros(N_sweep)
#    for i in range(N_sweep):
#        radius[i] = pool_result[i][0]
#        N_layer[i] = pool_result[i][0].size
#        RI[i] = pool_result[i][1]
#        Q_sca[i,:] = pool_result[i][2]
#        Q_abs[i,:] = pool_result[i][3]
#        Q_ext[i,:] = pool_result[i][4]
#        p[i,:,:,:] = pool_result[i][5]
#        cost[i] = pool_result[i][6]
    
    r_save_proc = np.zeros((data_size[comm.rank], int(np.max(N_layer))))
    n_re_proc = np.zeros((data_size[comm.rank], lam_plot.size, int(np.max(N_layer))+1))
    n_im_proc = np.zeros((data_size[comm.rank], lam_plot.size, int(np.max(N_layer))+1))
    for nr in range(data_size[comm.rank]):
        r_save_proc[nr,:int(N_layer_proc[nr])] = radius_proc[nr]
        n_re_proc[nr,:,:int(N_layer_proc[nr])+1] = np.real(RI_proc[nr])
        n_im_proc[nr,:,:int(N_layer_proc[nr])+1] = np.imag(RI_proc[nr])
    
    data_size_temp = data_size*int(np.max(N_layer))
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    r_temp = np.zeros(N_sweep*int(np.max(N_layer)))
    comm.Gatherv(r_save_proc.reshape(-1), [r_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    r_save = r_temp.reshape(N_sweep, int(np.max(N_layer)))
    
    data_size_temp = data_size*lam_plot.size*(int(np.max(N_layer)) + 1)
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    n_re_temp = np.zeros(N_sweep*lam_plot.size*(int(np.max(N_layer)) + 1))
    n_im_temp = np.zeros(N_sweep*lam_plot.size*(int(np.max(N_layer)) + 1))
    comm.Gatherv(n_re_proc.reshape(-1), [n_re_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(n_im_proc.reshape(-1), [n_im_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    n_save = (n_re_temp + 1j*n_im_temp).reshape(N_sweep, lam_plot.size, int(np.max(N_layer))+1)
    
    if comm.rank == 0:
        # Remove Designs that are Too Small
        filter_mask = r_save[:,0] > d_low
        r_save = r_save[filter_mask,:]
        n_save = n_save[filter_mask,:]
        Q_sca = Q_sca[filter_mask,:]
        Q_abs = Q_abs[filter_mask,:]
        Q_ext = Q_ext[filter_mask,:]
        p = p[filter_mask,:,:,:]
        diff_CS = diff_CS[filter_mask,:,:,:]
        N_layer = N_layer[filter_mask]
        cost = cost[filter_mask]
        
        # Sort from Best to Worst
        cost_sort = np.argsort(cost)
        cost = cost[cost_sort]
        r_save = r_save[cost_sort,:]
        n_save = n_save[cost_sort,:,:]
        Q_sca = Q_sca[cost_sort,:]
        Q_abs = Q_abs[cost_sort,:]
        Q_ext = Q_ext[cost_sort,:]
        p = p[cost_sort,:,:,:]
        diff_CS = diff_CS[cost_sort,:,:,:]
        N_layer = N_layer[cost_sort]
            
        np.savez(directory[:-9] + '/results/topopt_result_data_' + identifier, r=r_save, n=n_save,
                 Q_sca=Q_sca, Q_abs=Q_abs, Q_ext=Q_ext, p=p, diff_CS=diff_CS, N_layer=N_layer,
                 Q_sca_con=Q_sca_con, Q_abs_con=Q_abs_con, Q_ext_con=Q_ext_con, p_con=p_con, diff_CS_con=diff_CS_con,
                 d_low=d_low, r_max=r_max, cost=cost)

if __name__ == '__main__':
    if comm.rank == 0:
        if not os.path.isdir(directory[:-9] + "/debug"):
            os.mkdir(directory[:-9] + "/debug")
        if not os.path.isdir(directory[:-9] + "/results"):
            os.mkdir(directory[:-9] + "/results")
    
    comm.barrier()

    T1 = time.time()
    radius_sweep(identifier='theta160_phi0_lambda450_rmax1000', r_min=10, r_max=1000, N_sweep=991, d_low=5, max_layers=None)
    T2 = time.time()
    if comm.rank == 0:
        print('\nTotal time: ' + str(T2 - T1))