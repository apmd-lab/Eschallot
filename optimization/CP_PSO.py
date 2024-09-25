import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-13])

import numpy as np
import runfiles.CP_PSO_merit_fct as mf
import matplotlib.pyplot as plt
import time

result_file_name = 'MC'

def CP_PSO(mode, swarm_size, var_range, N_disc, N_choice, iteration_limit, stop_limit=100, c1=3, c2=2, w=0.8, starting_iteration=1, stop_count=0):
    """ mode: searching for 'min' or 'max'
        swarm_size: should be divisible by 10
        var_range: for continuous variables (dimension x 2 -> min, max)
        N_disc: number of discrete variables
        N_choice: number of choices for each discrete variable (e.g. if the discrete variable is the material for components 1 & 2, and there are 3 material candidates,
                  then N_disc=2, N_choice=3)
        iteration_limit: max number of total iterations
        stop_limit: max number of iterations without improvement
        c1: cognitive coefficient list->[continuous, discrete] (pulls towards personal best)
        c2: social coefficient list->[continuous, discrete] (pulls towards global best)
        w: inertia list->[continuous, discrete]
        starting_iteration: use when CP_PSO was stopped prematurely and you wish to start where the optimization left off
        stop_count: set stop_count when CP_PSO was stopped prematurely
        """

    if not os.path.isdir(directory + '/data'):
        os.mkdir(directory + '/data')
    if not os.path.isdir(directory + '/logs'):
        os.mkdir(directory + '/logs')
    
    iteration_count = starting_iteration
    dimension = np.shape(var_range)[0]
    vmax = 0.4*np.abs(var_range[:,1] - var_range[:,0])
    vmax_disc = 0.1
    v = np.zeros((dimension, swarm_size))
    index = np.zeros((dimension, swarm_size))
    
    #Particle Initialization
    if starting_iteration > 1:
        index = np.load(directory + '/data/' + result_file_name + '_particle_index.npy')
        v = np.load(directory + '/data/' + result_file_name + '_particle_v.npy')
        prob_disc = np.load(directory + '/data/' + result_file_name + '_prob_disc.npy')
        index_disc = np.load(directory + '/data/' + result_file_name + '_index_disc.npy')
        v_disc = np.load(directory + '/data/' + result_file_name + '_v_disc.npy')
        
        pbest_val = np.load(directory + '/data/' + result_file_name + '_pbest_val.npy')
        pbest_ind = np.load(directory + '/data/' + result_file_name + '_pbest_ind.npy')
        pbest_prob_disc = np.load(directory + '/data/' + result_file_name + '_pbest_prob_disc.npy')
        pbest_ind_disc = np.load(directory + '/data/' + result_file_name + '_pbest_ind_disc.npy')
        
        gbest_val = np.load(directory + '/data/' + result_file_name + '_gbest_val.npy')
        gbest_ind = np.load(directory + '/data/' + result_file_name + '_gbest_ind.npy')
        gbest_prob_disc = np.load(directory + '/data/' + result_file_name + '_gbest_prob_disc.npy')
        gbest_ind_disc = np.load(directory + '/data/' + result_file_name + '_gbest_ind_disc.npy')
    else:
        #Latin Hypercube Method
        increment = (var_range[:,1] - var_range[:,0])/swarm_size
        for dim in range(dimension):
            swarm_ind = np.random.permutation(swarm_size).reshape((1, swarm_size))
            if var_range[dim,0] == var_range[dim,1]:
                index[dim,:] = var_range[dim,0]*np.ones((1, swarm_size))
                v[dim,:] = np.zeros((1, swarm_size))
            else:
                index[dim,:] = var_range[dim,0] + (swarm_ind+1)*increment[dim] - increment[dim]*np.random.rand(1, swarm_size)
                v[dim,:] = -vmax[dim] + 2*vmax[dim]*np.random.rand(1, swarm_size)

        prob_disc = np.ones((N_disc, N_choice, swarm_size))/N_choice
        index_disc = np.zeros((N_disc, N_choice, swarm_size))
        for nd in range(N_disc):
            for s in range(swarm_size):
                n_choice = np.random.choice(np.linspace(0, N_choice, N_choice, endpoint=False).astype(int), p=prob_disc[nd,:,s])
                index_disc[nd,n_choice,s] = 1
        v_disc = -vmax_disc + 2*vmax_disc*np.random.rand(N_disc, N_choice, swarm_size)

        pbest_val = np.zeros((1, swarm_size))
        pbest_ind = np.zeros((dimension, swarm_size))
        pbest_prob_disc = np.ones((N_disc, N_choice, swarm_size))/N_choice
        pbest_ind_disc = np.zeros((N_disc, N_choice, swarm_size))
        
        gbest_val = 0
        gbest_ind = np.zeros((dimension, 1))
        gbest_prob_disc = np.ones((N_disc, N_choice, 1))/N_choice
        gbest_ind_disc = np.zeros((N_disc, N_choice, 1))
        
    valF = np.zeros(swarm_size)

    #Text File for Evaluation Status
    with open(directory + '/logs/PSO_status_' + result_file_name + '.txt', 'w') as f:
        f.write('-----\n')

    while True:
        if iteration_count > iteration_limit or stop_count >= stop_limit:
            FoM = mf.FoM(gbest_ind[:,-1], gbest_ind_disc[:,:,-1], pht_per_wvl=1e4)
            break
        
        # Particle Statistics
        for dim in range(dimension):
            fig, ax = plt.subplots(dpi=100)
            ax.hist(index[dim,:], bins=100, density=True)
            plt.savefig(directory + '/plots/PSO_stats_index_' + str(dim) + '.png', dpi=100)
            plt.close()
        
        for dim in range(N_disc):
            fig, ax = plt.subplots(dpi=100)
            for nc in range(N_choice):
                ax.hist(prob_disc[dim,nc,:], bins=100, density=True, alpha=0.5)
            plt.savefig(directory + '/plots/PSO_stats_prob_disc_' + str(dim) + '.png', dpi=100)
            plt.close()
        
        with open(directory + '/logs/PSO_status_' + result_file_name + '.txt', 'a') as f:
            f.write('Iteration %d\n' %iteration_count)
        
        for s in range(swarm_size):
            valF[s] = mf.FoM(index[:,s], index_disc[:,:,s])
        
        #Particle Best
        for s in range(swarm_size):
            if iteration_count == 1:
                pbest_val[0,s] = valF[s]
                pbest_ind[:,s] = index[:,s]
                pbest_prob_disc[:,:,s] = prob_disc[:,:,s]
                pbest_ind_disc[:,:,s] = index_disc[:,:,s]
            elif mode == 'min':
                if valF[s] < pbest_val[0,s]:
                    pbest_val[0,s] = valF[s]
                    pbest_ind[:,s] = index[:,s]
                    pbest_prob_disc[:,:,s] = prob_disc[:,:,s]
                    pbest_ind_disc[:,:,s] = index_disc[:,:,s]
            elif mode == 'max':
                if valF[s] > pbest_val[0,s]:
                    pbest_val[0,s] = valF[s]
                    pbest_ind[:,s] = index[:,s]
                    pbest_prob_disc[:,:,s] = prob_disc[:,:,s]
                    pbest_ind_disc[:,:,s] = index_disc[:,:,s]
            
        #Global Best
        if mode == 'min':
            new_Gbest = np.min(pbest_val)
            new_Gbest_pos = np.argmin(pbest_val)
        elif mode == 'max':
            new_Gbest = np.max(pbest_val)
            new_Gbest_pos = np.argmax(pbest_val)
        new_Gbest_ind = pbest_ind[:,new_Gbest_pos].reshape(dimension,1)
        new_Gbest_prob_disc = pbest_prob_disc[:,:,new_Gbest_pos].reshape(N_disc, N_choice, 1)
        new_Gbest_ind_disc = pbest_ind_disc[:,:,new_Gbest_pos].reshape(N_disc, N_choice, 1)
        
        if iteration_count == 1:
            gbest_val = new_Gbest
            gbest_ind = new_Gbest_ind
            gbest_prob_disc = new_Gbest_prob_disc
            gbest_ind_disc = new_Gbest_ind_disc
        elif iteration_count == 2:
            if new_Gbest == gbest_val:
                stop_count += 1
                gbest_val = np.append(gbest_val, new_Gbest)
                gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
            elif mode == 'min':
                if new_Gbest < gbest_val:
                    stop_count = 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                    gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                    gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
            elif mode == 'max':
                if new_Gbest > gbest_val:
                    stop_count = 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                    gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                    gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
        else:
            if new_Gbest == gbest_val[-1]:
                stop_count += 1
                gbest_val = np.append(gbest_val, new_Gbest)
                gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
            elif mode == 'min':
                if new_Gbest < gbest_val[-1]:
                    stop_count = 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                    gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                    gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
            elif mode == 'max':
                if new_Gbest > gbest_val[-1]:
                    stop_count = 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                    gbest_prob_disc = np.concatenate((gbest_prob_disc, new_Gbest_prob_disc), axis=2)
                    gbest_ind_disc = np.concatenate((gbest_ind_disc, new_Gbest_ind_disc), axis=2)
                    
        with open(directory + '/logs/PSO_status_' + result_file_name + '.txt', 'a') as f:
            f.write('Global Best Index:\t')
            for dim in range(dimension):
                f.write('%f\t' %gbest_ind[dim,-1])
            f.write('\n')
            for N in range(N_disc):
                f.write('%d\t' %np.argwhere(gbest_ind_disc[N,:,-1]==1)[0][0])
            if np.size(gbest_val) == 1:
                f.write('\nGlobal Best Value: %f\n\n' %gbest_val)
            else:
                f.write('\nGlobal Best Value: %f\n\n' %gbest_val[-1])
            
        #Parameter Updates
        if stop_count >= 5:
            w[0] = w[0]*0.99
            w[1] = w[1]*0.99
            vmax = vmax*0.99
        
        #Velocity Updates
        for s in range(swarm_size):
            v[:,s] = w[0]*v[:,s] + c1[0]*np.random.rand()*(pbest_ind[:,s] - index[:,s]) + c2[0]*np.random.rand()*(gbest_ind[:,-1] - index[:,s])
            v_disc[:,:,s] = w[1]*v_disc[:,:,s] + c1[1]*np.random.rand()*pbest_ind_disc[:,:,s] + c2[1]*np.random.rand()*gbest_ind_disc[:,:,-1]
            for dim in range(dimension):
                if np.abs(v[dim,s]) > vmax[dim]:
                    v[:,s] = v[:,s]*vmax[dim]/np.abs(v[dim,s]) #reduce magnitude while maintaining direction
            for N in range(N_disc):
                for nc in range(N_choice):
                    if np.abs(v_disc[N,nc,s]) > vmax_disc:
                        v_disc[:,:,s] = v_disc[:,:,s]*vmax_disc/np.abs(v_disc[N,nc,s])
        
        #Craziness
        N_cr = np.random.permutation(swarm_size)[:int(swarm_size/10)]
        if np.random.rand() < 0.22:
            for t in range(int(swarm_size/10)):
                v[:,N_cr[t]] = -vmax + vmax*np.random.rand(dimension)
                v_disc[:,:,N_cr[t]] = -vmax_disc + 2*vmax_disc*np.random.rand(N_disc, N_choice)
                
        #Position Updates
        for s in range(swarm_size):
            for dim in range(dimension):
                index[dim,s] += v[dim,s]
                if index[dim,s] < var_range[dim,0]:
                    index[dim,s] = var_range[dim,0] + (var_range[dim,0] - index[dim,s])
                elif index[dim,s] > var_range[dim,1]:
                    index[dim,s] = var_range[dim,1] - (index[dim,s] - var_range[dim,1])
            
            for N in range(N_disc):
                for nc in range(N_choice):
                    prob_disc[N,nc,s] += v_disc[N,nc,s]
                    if prob_disc[N,nc,s] < 0:
                        prob_disc[N,nc,s] *= -1
                    elif prob_disc[N,nc,s] > 1:
                        prob_disc[N,nc,s] = 2 - prob_disc[N,nc,s]
                prob_disc[N,:,s] /= np.sum(prob_disc[N,:,s])
        
        #Discrete Variable Update
        index_disc = np.zeros((N_disc, N_choice, swarm_size))
        for nd in range(N_disc):
            for s in range(swarm_size):
                n_choice = np.random.choice(np.linspace(0, N_choice, N_choice, endpoint=False).astype(int), p=prob_disc[nd,:,s])
                index_disc[nd,n_choice,s] = 1
        
        #Write Results into Text File (optional)
        if iteration_count == 1:
            with open(directory + '/logs/PSO_global_' + result_file_name + '.txt', 'w') as f:
                f.write('Iteration %d:\t' % iteration_count)
                f.write(np.format_float_scientific(gbest_val))
                f.write('\t\t')
                for dim in range(dimension):
                    #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                    f.write('%f\t' % gbest_ind[dim,-1])
                f.write('\n')
                for N in range(N_disc):
                    f.write('%f\t' %np.argwhere(gbest_ind_disc[N,:,-1]==1)[0][0])
                f.write('\n')
        else:
            if os.path.exists(directory + '/logs/PSO_global_' + result_file_name + '.txt'):
                with open(directory + '/logs/PSO_global_' + result_file_name + '.txt', 'a') as f:
                    f.write('Iteration %d:\t' % iteration_count)
                    f.write(np.format_float_scientific(gbest_val[-1]))
                    f.write('\t\t')
                    for dim in range(dimension):
                        #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                        f.write('%f\t' % gbest_ind[dim,-1])
                    f.write('\n')
                    for N in range(N_disc):
                        f.write('%f\t' %np.argwhere(gbest_ind_disc[N,:,-1]==1)[0][0])
                    f.write('\n')
            else:
                with open(directory + '/logs/PSO_global_' + result_file_name + '.txt', 'w') as f:
                    f.write('Iteration %d:\t' % iteration_count)
                    f.write(np.format_float_scientific(gbest_val[-1]))
                    f.write('\t\t')
                    for dim in range(dimension):
                        #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                        f.write('%f\t' % gbest_ind[dim,-1])
                    f.write('\n')
                    for N in range(N_disc):
                        f.write('%f\t' %np.argwhere(gbest_ind_disc[N,:,-1]==1)[0][0])
                    f.write('\n')
        
        #Save Progress
        np.save(directory + '/data/' + result_file_name + '_particle_index.npy', index)
        np.save(directory + '/data/' + result_file_name + '_prob_disc.npy', prob_disc)
        np.save(directory + '/data/' + result_file_name + '_index_disc.npy', index_disc)
        np.save(directory + '/data/' + result_file_name + '_particle_v.npy', v)
        np.save(directory + '/data/' + result_file_name + '_v_disc.npy', v_disc)
        np.save(directory + '/data/' + result_file_name + '_pbest_val.npy', pbest_val)
        np.save(directory + '/data/' + result_file_name + '_pbest_ind.npy', pbest_ind)
        np.save(directory + '/data/' + result_file_name + '_pbest_prob_disc.npy', pbest_prob_disc)
        np.save(directory + '/data/' + result_file_name + '_pbest_ind_disc.npy', pbest_ind_disc)
        np.save(directory + '/data/' + result_file_name + '_gbest_val.npy', gbest_val)
        np.save(directory + '/data/' + result_file_name + '_gbest_ind.npy', gbest_ind)
        np.save(directory + '/data/' + result_file_name + '_gbest_prob_disc.npy', gbest_prob_disc)
        np.save(directory + '/data/' + result_file_name + '_gbest_ind_disc.npy', gbest_ind_disc)
        
        iteration_count += 1
