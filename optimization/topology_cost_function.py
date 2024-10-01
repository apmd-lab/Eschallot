import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-13])

import numpy as np
from numba import jit
import mie.tmm_mie as tmm
import mie.tmm_mie_derivatives as deriv_tmm

def jacobian(r,
             n,
             index,
             lam,
             Q_sca_con,
             Q_abs_con,
             Q_ext_con,
             p_con,
             diff_CS_con,
             ml_init,
             ):
             
    ml_init.update(r, n)
    Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
        dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam,
                                                                                               ml_init.theta,
                                                                                               ml_init.phi,
                                                                                               ml_init.lmax,
                                                                                               ml_init.k,
                                                                                               r,
                                                                                               n,
                                                                                               ml_init.psi,
                                                                                               ml_init.dpsi,
                                                                                               ml_init.d2psi,
                                                                                               ml_init.ksi,
                                                                                               ml_init.dksi,
                                                                                               ml_init.d2ksi,
                                                                                               ml_init.pi_l,
                                                                                               ml_init.tau_l,
                                                                                               ml_init.eta_tilde)
    
    nancheck = np.isnan(np.sum(t_El, axis=0)) + np.isnan(np.sum(dt_El, axis=(0,2)))
    lmax = None
    if np.sum(nancheck) > 0:
        lmax = np.min(np.argwhere(nancheck)) - 1
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam,
                                                                                                   ml_init.theta,
                                                                                                   ml_init.phi,
                                                                                                   ml_init.lmax,
                                                                                                   ml_init.k,
                                                                                                   r,
                                                                                                   n,
                                                                                                   ml_init.psi,
                                                                                                   ml_init.dpsi,
                                                                                                   ml_init.d2psi,
                                                                                                   ml_init.ksi,
                                                                                                   ml_init.dksi,
                                                                                                   ml_init.d2ksi,
                                                                                                   ml_init.pi_l,
                                                                                                   ml_init.tau_l,
                                                                                                   ml_init.eta_tilde)
    
    if np.sum(np.isnan(dQ_sca)) + np.sum(np.isnan(dQ_abs)) + np.sum(np.isnan(dQ_ext)) + np.sum(np.isnan(dp)) + np.sum(np.isnan(d_diff_CS)) > 0:
        r[-1] = np.max((0, r[-1]))
        for l in range(r.size-2, -1, -1):
            if r[l] < r[l+1]:
                r[l] = r[l+1]
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam,
                                                                                                   ml_init.theta,
                                                                                                   ml_init.phi,
                                                                                                   ml_init.lmax,
                                                                                                   ml_init.k,
                                                                                                   r,
                                                                                                   n,
                                                                                                   ml_init.psi,
                                                                                                   ml_init.dpsi,
                                                                                                   ml_init.d2psi,
                                                                                                   ml_init.ksi,
                                                                                                   ml_init.dksi,
                                                                                                   ml_init.d2ksi,
                                                                                                   ml_init.pi_l,
                                                                                                   ml_init.tau_l,
                                                                                                   ml_init.eta_tilde)
    
    jac = build_jacobian(Q_sca_con,
                         Q_abs_con,
                         Q_ext_con,
                         p_con,
                         diff_CS_con,
                         Q_sca,
                         Q_abs,
                         Q_ext,
                         p,
                         dQ_sca,
                         dQ_abs,
                         dQ_ext,
                         dp,
                         d_diff_CS,
                         r,
                         lam,
                         ml_init.theta,
                         ml_init.phi)
    
#    mf1 = merit_fct(r-1e-6, n, index, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, ml_init)
#    mf2 = merit_fct(r+1e-6, n, index, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, ml_init)
#    jac_fd = (mf2-mf1)/2e-6
#
#    np.savez(directory[:-13] + "/debug/jacobian", Q_sca=Q_sca, p=p, dQ_sca=dQ_sca, dp=dp, r=r, t_El=t_El, dt_El=dt_El, jac=jac, jac_fd=jac_fd, ksi=ml_init.ksi, psi=ml_init.psi, dksi=ml_init.dksi,
#             dpsi=ml_init.dpsi, d2ksi=ml_init.d2ksi, d2psi=ml_init.d2psi, Q_abs=Q_abs, Q_ext=Q_ext, dQ_abs=dQ_abs, dQ_ext=dQ_ext, d_diff_CS=d_diff_CS, t_Ml=t_Ml, dt_Ml=dt_Ml)
    
    return jac

@jit(nopython=True, cache=True)
def build_jacobian(Q_sca_con,
                   Q_abs_con,
                   Q_ext_con,
                   p_con,
                   diff_CS_con,
                   Q_sca,
                   Q_abs,
                   Q_ext,
                   p,
                   dQ_sca,
                   dQ_abs,
                   dQ_ext,
                   dp,
                   d_diff_CS,
                   r,
                   lam,
                   theta,
                   phi,
                   ):

    jac = np.zeros(r.size)
    for l in range(r.size):
        jac[l] += np.sum(2*(Q_sca - Q_sca_con[0,0,:])*dQ_sca[:,l]*Q_sca_con[0,1,:])
        jac[l] += np.sum(2*np.maximum((Q_sca - Q_sca_con[1,0,:]), np.zeros(lam.size))*dQ_sca[:,l]*Q_sca_con[1,1,:])
        jac[l] += np.sum(2*np.minimum((Q_sca - Q_sca_con[2,0,:]), np.zeros(lam.size))*dQ_sca[:,l]*Q_sca_con[2,1,:])
        jac[l] += np.sum(dQ_sca[:,l]*Q_sca_con[3,1,:])
        
        jac[l] += np.sum(2*(Q_abs - Q_abs_con[0,0,:])*dQ_abs[:,l]*Q_abs_con[0,1,:])
        jac[l] += np.sum(2*np.maximum((Q_abs - Q_abs_con[1,0,:]), np.zeros(lam.size))*dQ_abs[:,l]*Q_abs_con[1,1,:])
        jac[l] += np.sum(2*np.minimum((Q_abs - Q_abs_con[2,0,:]), np.zeros(lam.size))*dQ_abs[:,l]*Q_abs_con[2,1,:])
        jac[l] += np.sum(dQ_abs[:,l]*Q_abs_con[3,1,:])
        
        jac[l] += np.sum(2*(Q_ext - Q_ext_con[0,0,:])*dQ_ext[:,l]*Q_ext_con[0,1,:])
        jac[l] += np.sum(2*np.maximum((Q_ext - Q_ext_con[1,0,:]), np.zeros(lam.size))*dQ_ext[:,l]*Q_ext_con[1,1,:])
        jac[l] += np.sum(2*np.minimum((Q_ext - Q_ext_con[2,0,:]), np.zeros(lam.size))*dQ_ext[:,l]*Q_ext_con[2,1,:])
        jac[l] += np.sum(dQ_ext[:,l]*Q_ext_con[3,1,:])
        
        jac[l] += np.sum(2*(p - p_con[0,0,:,:,:])*dp[:,:,:,l]*p_con[0,1,:,:,:])
        jac[l] += np.sum(2*np.maximum(p - p_con[1,0,:,:,:], np.zeros((lam.size, theta.size, phi.size)))*dp[:,:,:,l]*p_con[1,1,:,:,:])
        jac[l] += np.sum(2*np.minimum(p - p_con[2,0,:,:,:], np.zeros((lam.size, theta.size, phi.size)))*dp[:,:,:,l]*p_con[2,1,:,:,:])
        jac[l] += np.sum(dp[:,:,:,l]*p_con[3,1,:,:,:])
        
        jac[l] += np.sum(diff_CS_con[3,1,:,:,:]*d_diff_CS[:,:,:,l])

    return jac

def merit_fct(r,
              n,
              index,
              lam,
              Q_sca_con,
              Q_abs_con,
              Q_ext_con,
              p_con,
              diff_CS_con,
              ml_init,
              ):
    """ Constraints must be in the form (2 x ...) where 0: target value, 1: weight
    """
    ml_init.update(r, n)
    Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
        S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam,
                                                          ml_init.theta,
                                                          ml_init.phi,
                                                          ml_init.lmax,
                                                          ml_init.k,
                                                          r,
                                                          n,
                                                          ml_init.psi,
                                                          ml_init.dpsi,
                                                          ml_init.ksi,
                                                          ml_init.dksi,
                                                          ml_init.pi_l,
                                                          ml_init.tau_l,
                                                          ml_init.eta_tilde)
    
    nancheck = np.isnan(np.sum(t_El, axis=0))
    if np.sum(nancheck) > 0:
        lmax = np.min(np.argwhere(nancheck)) - 1
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam,
                                                              ml_init.theta,
                                                              ml_init.phi,
                                                              ml_init.lmax,
                                                              ml_init.k,
                                                              r,
                                                              n,
                                                              ml_init.psi,
                                                              ml_init.dpsi,
                                                              ml_init.ksi,
                                                              ml_init.dksi,
                                                              ml_init.pi_l,
                                                              ml_init.tau_l,
                                                              ml_init.eta_tilde)
    
    res = build_residual(Q_sca_con,
                         Q_abs_con,
                         Q_ext_con,
                         p_con,
                         diff_CS_con,
                         Q_sca,
                         Q_abs,
                         Q_ext,
                         p,
                         diff_CS,
                         lam,
                         ml_init.theta,
                         ml_init.phi)

    #np.savez(directory[:-13] + "/debug/merit_fct", Q_sca=Q_sca, p=p, t_El=t_El, r=r, res=res, Q_abs=Q_abs, Q_ext=Q_ext, diff_CS=diff_CS, t_Ml=t_Ml)

    return res

@jit(nopython=True, cache=True)
def build_residual(Q_sca_con,
                   Q_abs_con,
                   Q_ext_con,
                   p_con,
                   diff_CS_con,
                   Q_sca,
                   Q_abs,
                   Q_ext,
                   p,
                   diff_CS,
                   lam,
                   theta,
                   phi):
                   
    res = 0
    
    res += np.sum(Q_sca_con[0,1,:]*(Q_sca - Q_sca_con[0,0,:])**2)
    res += np.sum(Q_sca_con[1,1,:]*np.maximum(Q_sca - Q_sca_con[1,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_sca_con[2,1,:]*np.minimum(Q_sca - Q_sca_con[2,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_sca_con[3,1,:]*Q_sca)
    
    res += np.sum(Q_abs_con[0,1,:]*(Q_abs - Q_abs_con[0,0,:])**2)
    res += np.sum(Q_abs_con[1,1,:]*np.maximum(Q_abs - Q_abs_con[1,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_abs_con[2,1,:]*np.minimum(Q_abs - Q_abs_con[2,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_abs_con[3,1,:]*Q_abs)
    
    res += np.sum(Q_ext_con[0,1,:]*(Q_ext - Q_ext_con[0,0,:])**2)
    res += np.sum(Q_ext_con[1,1,:]*np.maximum(Q_ext - Q_ext_con[1,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_ext_con[2,1,:]*np.minimum(Q_ext - Q_ext_con[2,0,:], np.zeros(lam.size))**2)
    res += np.sum(Q_ext_con[3,1,:]*Q_ext)
    
    res += np.sum(p_con[0,1,:,:,:]*(p - p_con[0,0,:,:,:])**2)
    res += np.sum(p_con[1,1,:,:,:]*np.maximum(p - p_con[1,0,:,:,:], np.zeros((lam.size, theta.size, phi.size)))**2)
    res += np.sum(p_con[2,1,:,:,:]*np.minimum(p - p_con[2,0,:,:,:], np.zeros((lam.size, theta.size, phi.size)))**2)
    res += np.sum(p_con[3,1,:,:,:]*p)
    
    res += np.sum(diff_CS_con[3,1,:,:,:]*diff_CS)

    return res