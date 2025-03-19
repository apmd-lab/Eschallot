import numpy as np
import eschallot.mie.tmm_mie as tmm
import eschallot.mie.special_functions as spec

def cost(r,
         n,
         index,
         lam,
         ml_init,
         cost_custom,
         ):

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
    
    res = cost_custom.cost(Q_sca,
                           Q_abs,
                           Q_ext,
                           p,
                           diff_CS)

    return res

def shape_gradient(r,
                   n,
                   index,
                   lam,
                   ml_init,
                   cost_custom,
                   ):
             
    ml_init.update(r, n)
    Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml,\
        dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = tmm.efficiency_shape_derivatives(lam,
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
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = tmm.efficiency_shape_derivatives(lam,
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
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml = tmm.efficiency_shape_derivatives(lam,
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
    
    jac = cost_custom.gradient(Q_sca,
                               Q_abs,
                               Q_ext,
                               p,
                               diff_CS,
                               dQ_sca,
                               dQ_abs,
                               dQ_ext,
                               dp,
                               d_diff_CS,
                               r)
    
    return jac
    
def topology_gradient(r_needle,
                      n_needle,
                      ban_needle,
                      Q_sca,
                      Q_abs,
                      Q_ext,
                      p,
                      diff_CS,
                      lam,
                      theta,
                      phi,
                      lmax,
                      k,
                      r,
                      n,
                      pi_l,
                      tau_l,
                      Tcu_El,
                      Tcu_Ml,
                      Tcl_El,
                      Tcl_Ml,
                      T11_El,
                      T21_El,
                      T11_Ml,
                      T21_Ml,
                      t_El,
                      t_Ml,
                      S1,
                      S2,
                      d_low,
                      cost_custom):
                              
    wvl = lam.size
    
    # Index the layer in which the needle is to be inserted
    boundary = 0 # boolean that indicates whether the needle is being inserted at a boundary
    if r_needle < 1:
        boundary = 1
    else:
        if r.size == 1:
            if np.abs(r_needle - r) < 1:
                boundary = 1
            else:
                d_needle = 0 # index of layer (from outside) in which the needle is being inserted
        else:
            for l in range(r.size):
                if np.abs(r_needle - r[l]) < 1:
                    boundary = 1
                    break
            if r_needle < r[-1]:
                d_needle = r.size - 1
            else:
                for l in range(r.size-1):
                    if r_needle < r[l] and r_needle > r[l+1]:
                        d_needle = l
                        break
    
    if boundary:
        return 0
    else:
        if np.array_equal(n[:,d_needle+1], n_needle): # if needle & layer material are equal
            return 0
        elif ban_needle[d_needle]:
            return 0
        else:
            xj = n[:,d_needle+1]*k*r_needle
            eta_tilde = n[:,d_needle+1]/n_needle
            xj_temp = np.expand_dims(xj, axis=1)

            ksi, dksi = spec.RB_ksi(xj_temp, lmax)
            psi, dpsi = spec.RB_psi(xj_temp, lmax)
            ksi = ksi[:,:,0]
            dksi = dksi[:,:,0]
            psi = psi[:,:,0]
            dpsi = dpsi[:,:,0]

            coeff = np.zeros((wvl, lmax+1))
            for n_l in range(lmax+1):
                coeff[:,n_l] = n_l*(n_l + 1)
            
            d2ksi = -(1 - coeff/xj_temp**2)*ksi
            d2psi = -(1 - coeff/xj_temp**2)*psi
            
            dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS = tmm.efficiency_topology_derivatives(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, diff_CS,
                                                                                                      lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde,
                                                                                                      Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2)
            
            nancheck = np.isnan(np.sum(dt_El, axis=0))
            lmax = None
            if np.sum(nancheck) > 0:
                lmax = np.min(np.argwhere(nancheck)[:,0]) - 1
    
                ksi, dksi = spec.RB_ksi(xj_temp, lmax)
                psi, dpsi = spec.RB_psi(xj_temp, lmax)
                ksi = ksi[:,:,0]
                dksi = dksi[:,:,0]
                psi = psi[:,:,0]
                dpsi = dpsi[:,:,0]
                    
                d2ksi = -(1 - coeff[:,:lmax+1]/xj_temp**2)*ksi
                d2psi = -(1 - coeff[:,:lmax+1]/xj_temp**2)*psi
                
                Tcu_El = Tcu_El[:,:lmax,:,:,:]
                Tcu_Ml = Tcu_Ml[:,:lmax,:,:,:]
                Tcl_El = Tcl_El[:,:lmax,:,:,:]
                Tcl_Ml = Tcl_Ml[:,:lmax,:,:,:]
                T11_El = T11_El[:,:lmax]
                T21_El = T21_El[:,:lmax]
                T11_Ml = T11_Ml[:,:lmax]
                T21_Ml = T21_Ml[:,:lmax]
                t_El = t_El[:,:lmax]
                t_Ml = t_Ml[:,:lmax]
                pi_l = pi_l[:,:lmax+1]
                tau_l = tau_l[:,:lmax+1]
                
                dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS = tmm.efficiency_topology_derivatives(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, diff_CS,
                                                                                                          lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde,
                                                                                                          Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2)
                
            dMF_val = cost_custom.gradient(Q_sca, Q_abs, Q_ext, p, diff_CS, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, r)
            
            return dMF_val