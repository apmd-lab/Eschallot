import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-11])

import numpy as np
from sympy import solve
from sympy.abc import x, a, b, c, d
import mpmath as mp
import matplotlib.pyplot as plt
import util.read_mat_data as rmd
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

mp.mp.dps = 100

def maxwell_garnett_mixing_2phase(f_vol, eps):
    eps_h = eps[:,0]
    eps_i = eps[:,1]
    a0 = (eps_i - eps_h)/(eps_i + 2*eps_h)
    
    eps_eff = eps_h*(1 + 2*f_vol*a0)/(1 - f_vol*a0)
    
    return eps_eff

# Refer to Jansson et al. "Selection of the physically correct solution in the n-media Bruggeman effective medium approximation", Optics Communications (1994).
def bruggeman_mixing_2phase(f_vol, eps, lam):
    """
    works for 2 phase mixture only
    eps: wvl x n_material
    """
    
    quadratic = a*x**2 + b*x + c
    sol = solve(quadratic, x, dict=True)
    
    eps_eff = np.zeros(eps.shape[0]).astype(np.complex128)
    for nl in range(eps.shape[0]):
        a0 = -2*np.sum(f_vol)
        b0 = 2*(f_vol[0]*eps[nl,0] + f_vol[1]*eps[nl,1]) - (f_vol[0]*eps[nl,1] + f_vol[1]*eps[nl,0])
        c0 = np.sum(f_vol)*np.prod(eps[nl,:])
    
        eps_root = np.zeros(len(sol)).astype(np.complex128)
        for nroot in range(len(sol)):
            eps_root[nroot] = complex(sol[nroot][x].subs([(a,a0),(b,b0),(c,c0)]).evalf())
        
        # Circle Test
        circle_test = np.zeros(len(sol)).astype(bool)
        for nroot in range(len(sol)):
            eps0 = mp.mpc(eps[nl,0])
            eps1 = mp.mpc(eps[nl,1])
            epsR = mp.mpc(eps_root[nroot])
        
            if mp.fabs(mp.conj(eps0)*eps1 - eps0*mp.conj(eps1)) == 0:
                transform0 = eps0*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                transform1 = eps1*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                sign = (-1)**(mp.re(transform1) < mp.re(transform0))
                transform0 *= sign
                transform1 *= sign
                transform = sign*eps_root[nroot]*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                circle_test[nroot] = (mp.im(transform) == 0)*(mp.re(transform) >= mp.re(transform0))*(mp.re(transform) <= mp.re(transform1))
            else:
                center = eps0*eps1*(mp.conj(eps0) - mp.conj(eps1))/(mp.conj(eps0)*eps1 - eps0*mp.conj(eps1))
                transform0 = (eps0 - center)/mp.fabs(center)*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                sign = mp.sign(mp.im(transform0))
                transform0 *= sign
                transform = sign*(eps_root[nroot] - center)/mp.fabs(center)*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                circle_test[nroot] = (mp.fabs(transform) <= 1)*(mp.im(transform) >= mp.im(transform0))
        
        try:
            eps_eff[nl] = eps_root[circle_test]
        except:
            eps_eff[nl] = np.nan
        
    if np.any(np.isnan(eps_eff)):
        nan_mask = np.isnan(eps_eff)
        eps_eff[nan_mask] = np.interp(lam[nan_mask], lam[~nan_mask], eps_eff[~nan_mask])
    
    return eps_eff

def multishell_particle_medium(n, r, N_layer, f_vol_particle):
    eps_eff = n[:,-1]**2
    for l in range(N_layer-1):
        f_vol = (r[-1-l]/r[-2-l])**3
        eps_list = np.vstack((n[:,-2+l]**2, eps_eff)).T
        eps_eff = maxwell_garnett_mixing(f_vol, eps_list)
    
    eps_list = np.vstack((n[:,0]**2, eps_eff)).T
    if f_vol_particle >= 0.1:
        eps_eff = bruggeman_mixing_2phase(f_vol_particle, eps_list)
    else:
        eps_eff = maxwell_garnett_mixing_2phase(f_vol_particle, eps_list)
    
    return eps_eff

def bruggeman_mixing_3phase(f_vol, eps, lam):
    """
    works for 3 phase mixture only
    eps: wvl x n_material
    """
    
    cubic = a*x**3 + b*x**2 + c*x + d
    sol = solve(cubic, x, dict=True)
    
    eps_eff = np.zeros(eps.shape[0]).astype(np.complex128)
    for nl in range(eps.shape[0]):
        a0 = -4*np.sum(f_vol)
        b0 = 2*(f_vol[0]*(2*eps[nl,0] - eps[nl,1]) + f_vol[1]*(2*eps[nl,1] - eps[nl,0]) + f_vol[2]*(2*eps[nl,2] - eps[nl,0]) - f_vol[0]*eps[nl,2] - f_vol[1]*eps[nl,2] - f_vol[2]*eps[nl,1])
        c0 = 2*(f_vol[0]*eps[nl,0]*eps[nl,1] + f_vol[1]*eps[nl,0]*eps[nl,1] + f_vol[2]*eps[nl,0]*eps[nl,2])\
             + (2*eps[nl,0] - eps[nl,1])*f_vol[0]*eps[nl,2]\
             + (2*eps[nl,1] - eps[nl,0])*f_vol[1]*eps[nl,2]\
             + (2*eps[nl,2] - eps[nl,0])*f_vol[2]*eps[nl,1]
        d0 = np.sum(f_vol)*np.prod(eps[nl,:])
    
        eps_root = np.zeros(len(sol)).astype(np.complex128)
        for nroot in range(len(sol)):
            eps_root[nroot] = complex(sol[nroot][x].subs([(a,a0),(b,b0),(c,c0),(d,d0)]).evalf())
    
        # Triangle Test
        triangle_test = np.zeros((3, len(sol)))
        for nroot in range(len(sol)):
            sign = np.sign(np.imag((eps[nl,2] - eps[nl,0])*np.conj(eps[nl,1] - eps[nl,0])))
            triangle_test[0,nroot] = np.imag(sign*(eps_root[nroot] - eps[nl,0])*np.conj(eps[nl,1] - eps[nl,0])) >= 0
            
            sign = np.sign(np.imag((eps[nl,0] - eps[nl,1])*np.conj(eps[nl,2] - eps[nl,1])))
            triangle_test[1,nroot] = np.imag(sign*(eps_root[nroot] - eps[nl,1])*np.conj(eps[nl,2] - eps[nl,1])) >= 0
            
            sign = np.sign(np.imag((eps[nl,1] - eps[nl,2])*np.conj(eps[nl,0] - eps[nl,2])))
            triangle_test[2,nroot] = np.imag(sign*(eps_root[nroot] - eps[nl,2])*np.conj(eps[nl,0] - eps[nl,2])) >= 0
        
        # Circle Test
        circle_test = np.zeros((3, len(sol)))
        for nroot in range(len(sol)):
            eps0 = mp.mpc(eps[nl,0])
            eps1 = mp.mpc(eps[nl,1])
            eps2 = mp.mpc(eps[nl,2])
            epsR = mp.mpc(eps_root[nroot])
        
            if mp.fabs(mp.conj(eps0)*eps1 - eps0*mp.conj(eps1)) == 0:
                transform0 = eps0*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                transform1 = eps1*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                sign = (-1)**(mp.re(transform1) < mp.re(transform0))
                transform0 *= sign
                transform1 *= sign
                transform = sign*eps_root[nroot]*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                circle_test[0,nroot] = (mp.im(transform) == 0)*(mp.re(transform) >= mp.re(transform0))*(mp.re(transform) <= mp.re(transform1))
            else:
                center = eps0*eps1*(mp.conj(eps0) - mp.conj(eps1))/(mp.conj(eps0)*eps1 - eps0*mp.conj(eps1))
                transform0 = (eps0 - center)/mp.fabs(center)*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                sign = mp.sign(mp.im(transform0))
                transform0 *= sign
                transform = sign*(eps_root[nroot] - center)/mp.fabs(center)*mp.conj(eps1 - eps0)/mp.fabs(eps1 - eps0)
                circle_test[0,nroot] = (mp.fabs(transform) <= 1)*(mp.im(transform) >= mp.im(transform0))
            
            if mp.fabs(mp.conj(eps1)*eps[nl,2] - eps1*mp.conj(eps[nl,2])) == 0:
                transform0 = eps1*mp.conj(eps[nl,2] - eps1)/mp.fabs(eps[nl,2] - eps1)
                transform1 = eps[nl,2]*mp.conj(eps[nl,2] - eps1)/mp.fabs(eps[nl,2] - eps1)
                sign = (-1)**(mp.re(transform1) < mp.re(transform0))
                transform0 *= sign
                transform1 *= sign
                transform = sign*eps_root[nroot]*mp.conj(eps[nl,2] - eps1)/mp.fabs(eps[nl,2] - eps1)
                circle_test[nroot] = (mp.im(transform) == 0)*(mp.re(transform) >= mp.re(transform0))*(mp.re(transform) <= mp.re(transform1))
            else:
                center = eps1*eps[nl,2]*(mp.conj(eps1) - mp.conj(eps[nl,2]))/(mp.conj(eps1)*eps[nl,2] - eps1*mp.conj(eps[nl,2]))
                transform0 = (eps1 - center)/mp.fabs(center)*mp.conj(eps[nl,2] - eps1)/mp.fabs(eps[nl,2] - eps1)
                sign = mp.sign(mp.im(transform0))
                transform0 *= sign
                transform = sign*(eps_root[nroot] - center)/mp.fabs(center)*mp.conj(eps[nl,2] - eps1)/mp.fabs(eps[nl,2] - eps1)
                circle_test[1,nroot] = (mp.fabs(transform) <= 1)*(mp.im(transform) >= mp.im(transform0))
            
            if mp.fabs(mp.conj(eps[nl,2])*eps0 - eps[nl,2]*mp.conj(eps0)) == 0:
                transform0 = eps[nl,2]*mp.conj(eps0 - eps[nl,2])/mp.fabs(eps0 - eps[nl,2])
                transform1 = eps0*mp.conj(eps0 - eps[nl,2])/mp.fabs(eps0 - eps[nl,2])
                sign = (-1)**(mp.re(transform1) < mp.re(transform0))
                transform0 *= sign
                transform1 *= sign
                transform = sign*eps_root[nroot]*mp.conj(eps0 - eps[nl,2])/mp.fabs(eps0 - eps[nl,2])
                circle_test[nroot] = (mp.im(transform) == 0)*(mp.re(transform) >= mp.re(transform0))*(mp.re(transform) <= mp.re(transform1))
            else:
                center = eps[nl,2]*eps0*(mp.conj(eps[nl,2]) - mp.conj(eps0))/(mp.conj(eps[nl,2])*eps0 - eps[nl,2]*mp.conj(eps0))
                transform0 = (eps[nl,2] - center)/mp.fabs(center)*mp.conj(eps0 - eps[nl,2])/mp.fabs(eps0 - eps[nl,2])
                sign = mp.sign(mp.im(transform0))
                transform0 *= sign
                transform = sign*(eps_root[nroot] - center)/mp.fabs(center)*mp.conj(eps0 - eps[nl,2])/mp.fabs(eps0 - eps[nl,2])
                circle_test[2,nroot] = (mp.fabs(transform) <= 1)*(mp.im(transform) >= mp.im(transform0))
        
        test_combined = np.all(triangle_test + circle_test, axis=0)
        try:
            eps_eff[nl] = eps_root[test_combined]
        except:
            eps_eff[nl] = np.nan
    
    if np.any(np.isnan(eps_eff)):
        nan_mask = np.isnan(eps_eff)
        eps_eff[nan_mask] = np.interp(lam[nan_mask], lam[~nan_mask], eps_eff[~nan_mask])
    
    return eps_eff

def multitype_particle_medium(lam, n, f_vol_particle):
    eps_list = n**2 # eps_incl1, eps_incl2, eps_host
    
    if np.array_equal(eps_list[:,0], eps_list[:,1]):
        eps_list = eps_list[:,1:]
        f_vol_particle = np.array([1-f_vol_particle[-1],f_vol_particle[-1]])
        
        eps_eff = bruggeman_mixing_2phase(f_vol_particle, eps_list, lam)
    else:
        eps_eff = bruggeman_mixing_3phase(f_vol_particle, eps_list, lam)
    
    # PDF
    if rank == 0:
        if not os.path.isdir(directory + '/plots'):
            os.mkdir(directory + '/plots')
    
        n_eff = np.sqrt(eps_eff)
    
        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
        ax.plot(lam, np.real(n[:,0]), linewidth=2, color='firebrick', label='Component 1')
        ax.plot(lam, np.real(n[:,1]), linewidth=2, color='goldenrod', label='Component 2')
        ax.plot(lam, np.real(n[:,2]), linewidth=2, color='royalblue', label='Component 3')
        ax.plot(lam, np.imag(n[:,0]), linewidth=2, color='firebrick', linestyle='dashed')
        ax.plot(lam, np.imag(n[:,1]), linewidth=2, color='goldenrod', linestyle='dashed')
        ax.plot(lam, np.imag(n[:,2]), linewidth=2, color='royalblue', linestyle='dashed')
        ax.plot(lam, np.real(n_eff), linewidth=2, color='black', label='Eff. Medium')
        ax.plot(lam, np.imag(n_eff), linewidth=2, color='black', linestyle='dashed')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Refractive Index')
        ax.legend()
        plt.savefig(directory + '/plots/effective_medium')
        plt.close()
    
    return eps_eff
