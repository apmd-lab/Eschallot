import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-4])

import numpy as np
from numba import jit
import mie.tmm_mie as tmm
import mie.special_functions as spec

# @jit(nopython=True, cache=True)
def efficiency_derivatives(lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size
    ph = phi.size
    
    Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml,\
        T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml = tmm.calculate_tmm_matrix(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)

    # Cross sections
    k_ext = np.real(n[:,0]*k) # wvl
    coeff_C = np.zeros((wvl, lmax))
    for n_l in range(1, lmax+1):
        coeff_C[:,n_l-1] = (2*n_l + 1)
    
    C_sca = coeff_C*(np.abs(t_El)**2 + np.abs(t_Ml)**2) # wvl x lmax
    C_abs = coeff_C*(2 - np.abs(1 + 2*t_El)**2 - np.abs(1 + 2*t_Ml)**2)
    C_ext = coeff_C*(np.real(t_El) + np.real(t_Ml))

    C_sca = (2*np.pi/k_ext**2)*np.sum(C_sca, axis=1) # wvl
    C_abs = (np.pi/(2*k_ext**2))*np.sum(C_abs, axis=1)
    C_ext = (-2*np.pi/k_ext**2)*np.sum(C_ext, axis=1)
    Q_sca = C_sca/(np.pi*r[0]**2)
    Q_abs = C_abs/(np.pi*r[0]**2)
    Q_ext = C_ext/(np.pi*r[0]**2)
    
    # Scattering amplitudes
    t_El_temp = np.expand_dims(t_El, axis=1)
    t_Ml_temp = np.expand_dims(t_Ml, axis=1)
    pi_l = np.expand_dims(pi_l, axis=0) # wvl x th x lmax
    tau_l = np.expand_dims(tau_l, axis=0)
    
    coeff_S = np.zeros((wvl, th, lmax))
    for n_l in range(1, lmax+1):
        coeff_S[:,:,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))

    S1 = coeff_S*(t_El_temp*pi_l[:,:,1:] + t_Ml_temp*tau_l[:,:,1:]) # wvl x th x lmax
    S2 = coeff_S*(t_El_temp*tau_l[:,:,1:] + t_Ml_temp*pi_l[:,:,1:])
    S1 = np.sum(S1, axis=2) # wvl x th
    S2 = np.sum(S2, axis=2)

    # Scattering angular distributions
    S1_temp = np.expand_dims(S1, axis=2) # wvl x th x ph
    S2_temp = np.expand_dims(S2, axis=2)
    
    sin_phi = np.sin(phi)
    sin_phi = np.expand_dims(np.expand_dims(sin_phi, axis=0), axis=0) # wvl x th x ph
    cos_phi = np.cos(phi)
    cos_phi = np.expand_dims(np.expand_dims(cos_phi, axis=0), axis=0)
    
    k_ext_temp = np.expand_dims(np.expand_dims(k_ext, axis=1), axis=2)
    C_sca_temp = np.expand_dims(np.expand_dims(C_sca, axis=1), axis=2)
    
    diff_CS = (np.abs(S1_temp)**2*sin_phi**2 + np.abs(S2_temp)**2*cos_phi**2)/k_ext_temp**2 # wvl x th x ph
    p = diff_CS/C_sca_temp

    # Change in matrix quantities
    dTj_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    dTj_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    dT_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    dT_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    
    n_temp = np.expand_dims(n, axis=1)
    k_temp = np.expand_dims(np.expand_dims(k, axis=1), axis=2)
    eta_tilde_temp = np.expand_dims(eta_tilde, axis=1)

    dTj_El[:,:,:,0,0] = -1j*(n_temp[:,:,:-1]*k_temp*d2ksi[1,:,1:,:]*psi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dksi[1,:,1:,:]*dpsi[0,:,1:,:])\
        + (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*dksi[1,:,1:,:]*dpsi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*ksi[1,:,1:,:]*d2psi[0,:,1:,:])
    dTj_El[:,:,:,0,1] = -1j*(n_temp[:,:,:-1]*k_temp*d2ksi[1,:,1:,:]*ksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dksi[1,:,1:,:]*dksi[0,:,1:,:])\
        + (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*dksi[1,:,1:,:]*dksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*ksi[1,:,1:,:]*d2ksi[0,:,1:,:])
    dTj_El[:,:,:,1,0] = 1j*(n_temp[:,:,:-1]*k_temp*d2psi[1,:,1:,:]*psi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dpsi[1,:,1:,:]*dpsi[0,:,1:,:])\
        - (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*dpsi[1,:,1:,:]*dpsi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*psi[1,:,1:,:]*d2psi[0,:,1:,:])
    dTj_El[:,:,:,1,1] = 1j*(n_temp[:,:,:-1]*k_temp*d2psi[1,:,1:,:]*ksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dpsi[1,:,1:,:]*dksi[0,:,1:,:])\
        - (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*dpsi[1,:,1:,:]*dksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*psi[1,:,1:,:]*d2ksi[0,:,1:,:])
        
    dTj_Ml[:,:,:,0,0] = -(1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*d2ksi[1,:,1:,:]*psi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dksi[1,:,1:,:]*dpsi[0,:,1:,:])\
        + 1j*(n_temp[:,:,:-1]*k_temp*dksi[1,:,1:,:]*dpsi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*ksi[1,:,1:,:]*d2psi[0,:,1:,:])
    dTj_Ml[:,:,:,0,1] = -(1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*d2ksi[1,:,1:,:]*ksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dksi[1,:,1:,:]*dksi[0,:,1:,:])\
        + 1j*(n_temp[:,:,:-1]*k_temp*dksi[1,:,1:,:]*dksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*ksi[1,:,1:,:]*d2ksi[0,:,1:,:])
    dTj_Ml[:,:,:,1,0] = (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*d2psi[1,:,1:,:]*psi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dpsi[1,:,1:,:]*dpsi[0,:,1:,:])\
        - 1j*(n_temp[:,:,:-1]*k_temp*dpsi[1,:,1:,:]*dpsi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*psi[1,:,1:,:]*d2psi[0,:,1:,:])
    dTj_Ml[:,:,:,1,1] = (1j/eta_tilde_temp)*(n_temp[:,:,:-1]*k_temp*d2psi[1,:,1:,:]*ksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*dpsi[1,:,1:,:]*dksi[0,:,1:,:])\
        - 1j*(n_temp[:,:,:-1]*k_temp*dpsi[1,:,1:,:]*dksi[0,:,1:,:] + n_temp[:,:,1:]*k_temp*psi[1,:,1:,:]*d2ksi[0,:,1:,:])
    
    for l in range(layer):
        if l == 0 and l == layer-1:
            dT_El[:,:,l,:,:] = dTj_El[:,:,l,:,:].copy()
            dT_Ml[:,:,l,:,:] = dTj_Ml[:,:,l,:,:].copy()
        elif l == 0:
            dT_El[:,:,l,:,:] = dTj_El[:,:,l,:,:]@Tcl_El[:,:,l+1,:,:]
            dT_Ml[:,:,l,:,:] = dTj_Ml[:,:,l,:,:]@Tcl_Ml[:,:,l+1,:,:]
        elif l == layer-1:
            dT_El[:,:,l,:,:] = Tcu_El[:,:,l-1,:,:]@dTj_El[:,:,l,:,:]
            dT_Ml[:,:,l,:,:] = Tcu_Ml[:,:,l-1,:,:]@dTj_Ml[:,:,l,:,:]
        else:
            dT_El[:,:,l,:,:] = Tcu_El[:,:,l-1,:,:]@dTj_El[:,:,l,:,:]@Tcl_El[:,:,l+1,:,:]
            dT_Ml[:,:,l,:,:] = Tcu_Ml[:,:,l-1,:,:]@dTj_Ml[:,:,l,:,:]@Tcl_Ml[:,:,l+1,:,:]
    
    dT11_El = dT_El[:,:,:,0,0]
    dT21_El = dT_El[:,:,:,1,0]
    dT11_Ml = dT_Ml[:,:,:,0,0]
    dT21_Ml = dT_Ml[:,:,:,1,0]
    
    T11_El_temp = np.expand_dims(T11_El, axis=2)
    T21_El_temp = np.expand_dims(T21_El, axis=2)
    T11_Ml_temp = np.expand_dims(T11_Ml, axis=2)
    T21_Ml_temp = np.expand_dims(T21_Ml, axis=2)

    dt_El = (1/T11_El_temp**2)*(T11_El_temp*dT21_El - T21_El_temp*dT11_El) # wvl x lmax x layer
    dt_Ml = (1/T11_Ml_temp**2)*(T11_Ml_temp*dT21_Ml - T21_Ml_temp*dT11_Ml)
    
    # Efficiencies
    coeff_C = np.expand_dims(coeff_C, axis=2)
    t_El_temp = np.expand_dims(t_El, axis=2)
    t_Ml_temp = np.expand_dims(t_Ml, axis=2)
    k_ext_temp = np.expand_dims(np.expand_dims(k_ext, axis=1), axis=2) # wvl x lmax x layer
    
    dC_sca = 2*coeff_C*np.real(np.conj(t_El_temp)*dt_El + np.conj(t_Ml_temp)*dt_Ml)
    dC_abs = 4*coeff_C*np.real((1 + 2*np.conj(t_El_temp))*dt_El + (1 + 2*np.conj(t_Ml_temp))*dt_Ml)
    dC_ext = coeff_C*np.real(dt_El + dt_Ml)
    
    dC_sca_out = dC_sca*2*np.pi/k_ext_temp**2 # wvl x lmax x layer
    dC_sca *= 2/(k_ext_temp**2*r[0]**2)
    dC_abs *= -1/(2*k_ext_temp**2*r[0]**2)
    dC_ext *= -2/(k_ext_temp**2*r[0]**2)
    
    dC_sca_out = np.sum(dC_sca_out, axis=1) # wvl x layer
    dQ_sca = np.sum(dC_sca, axis=1)
    dQ_abs = np.sum(dC_abs, axis=1)
    dQ_ext = np.sum(dC_ext, axis=1)
    dQ_sca[:,0] += (-2/(np.pi*r[0]**3))*C_sca
    dQ_abs[:,0] += (-2/(np.pi*r[0]**3))*C_abs
    dQ_ext[:,0] += (-2/(np.pi*r[0]**3))*C_ext
    
    # Scattering Matrix
    coeff_S = np.expand_dims(coeff_S, axis=3)
    dt_El_temp = np.expand_dims(dt_El, axis=1)
    dt_Ml_temp = np.expand_dims(dt_Ml, axis=1)
    pi_l = np.expand_dims(pi_l, axis=3)
    tau_l = np.expand_dims(tau_l, axis=3)

    dS1 = coeff_S*(dt_El_temp*pi_l[:,:,1:,:] + dt_Ml_temp*tau_l[:,:,1:,:]) # wvl x th x lmax x layer
    dS2 = coeff_S*(dt_El_temp*tau_l[:,:,1:,:] + dt_Ml_temp*pi_l[:,:,1:,:])
    
    dS1 = np.sum(dS1, axis=2) # wvl x th x layer
    dS2 = np.sum(dS2, axis=2)
    
    # Angular Power Distribution
    k_ext_temp = np.expand_dims(np.expand_dims(np.expand_dims(k_ext, axis=1), axis=2), axis=3) # wvl x th x ph x layer
    S1_temp = np.expand_dims(S1_temp, axis=3)
    S2_temp = np.expand_dims(S2_temp, axis=3)
    dS1_temp = np.expand_dims(dS1, axis=2)
    dS2_temp = np.expand_dims(dS2, axis=2)
    sin_phi = np.expand_dims(sin_phi, axis=3)
    cos_phi = np.expand_dims(cos_phi, axis=3)
    C_sca = np.expand_dims(np.expand_dims(np.expand_dims(C_sca, axis=1), axis=2), axis=3)
    dC_sca_out = np.expand_dims(np.expand_dims(dC_sca_out, axis=1), axis=2)
    
    d_diff_CS = (2/(k_ext_temp**2))*(np.real(np.conj(S1_temp)*dS1_temp)*sin_phi**2 + np.real(np.conj(S2_temp)*dS2_temp)*cos_phi**2) # wvl x th x ph x layer
    dp = -(1/(k_ext_temp**2*C_sca**2))*(np.abs(S1_temp)**2*sin_phi**2 + np.abs(S2_temp)**2*cos_phi**2)*dC_sca_out\
        + (2/(k_ext_temp**2*C_sca))*(np.real(np.conj(S1_temp)*dS1_temp)*sin_phi**2 + np.real(np.conj(S2_temp)*dS2_temp)*cos_phi**2)
    
    return Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml

def needle_derivative_wrapper(r_needle,
                              n_needle,
                              ban_needle,
                              Q_sca,
                              Q_abs,
                              Q_ext,
                              p,
                              Q_sca_con,
                              Q_abs_con,
                              Q_ext_con,
                              p_con,
                              diff_CS_con,
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
                              d_low):
                              
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
            
            dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS = needle_derivative(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con,
                                                                                    lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde,
                                                                                    Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2)
            
            nancheck = np.isnan(np.sum(dt_El, axis=0))
            lmax = None
            if np.sum(nancheck) > 0:
                lmax = np.min(np.argwhere(nancheck)) - 1
    
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
                
                dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS = needle_derivative(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con,
                                                                                        lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde,
                                                                                        Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2)
                
            dMF_val = build_topology_nucleation_derivative_directional(Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, Q_sca, Q_abs, Q_ext, p,
                                                                       dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, lam, theta, phi)
            
            return dMF_val

# @jit(nopython=True, cache=True)
def needle_derivative(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con,
                      lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde,
                      Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2):    
    wvl = lam.size
    layer = r.size
    th = theta.size
    ph = phi.size

    kj = n[:,d_needle+1]*k
    kn = n_needle*k
    kj = np.expand_dims(kj, axis=1)
    kn = np.expand_dims(kn, axis=1)
    eta_tilde = np.expand_dims(eta_tilde, axis=1)
    
    c2_El = (kj - kn/eta_tilde)
    c2_Ml = (kj - kn*eta_tilde)
    xn = kn*r_needle
    
    coeff = np.zeros((wvl, lmax))
    for n_l in range(1, lmax+1):
        coeff[:,n_l-1] = n_l*(n_l + 1)
    
    c3_El = (1 - coeff/xn**2)*kn*eta_tilde
    c3_Ml = (1 - coeff/xn**2)*kn/eta_tilde
    
    dTj_El = np.zeros((wvl, lmax, 2, 2)).astype(np.complex128)
    dTj_Ml = np.zeros((wvl, lmax, 2, 2)).astype(np.complex128)
    
    dTj_El[:,:,0,0] = -1j*kj*d2ksi[:,1:]*psi[:,1:] + 1j*c2_El*dksi[:,1:]*dpsi[:,1:] - 1j*c3_El*ksi[:,1:]*psi[:,1:]
    dTj_El[:,:,0,1] = -1j*kj*d2ksi[:,1:]*ksi[:,1:] + 1j*c2_El*dksi[:,1:]**2 - 1j*c3_El*ksi[:,1:]**2
    dTj_El[:,:,1,0] = 1j*kj*d2psi[:,1:]*psi[:,1:] - 1j*c2_El*dpsi[:,1:]**2 + 1j*c3_El*psi[:,1:]**2
    dTj_El[:,:,1,1] = 1j*kj*d2psi[:,1:]*ksi[:,1:] - 1j*c2_El*dksi[:,1:]*dpsi[:,1:] + 1j*c3_El*ksi[:,1:]*psi[:,1:]
    
    dTj_Ml[:,:,0,0] = -1j*kj*d2ksi[:,1:]*psi[:,1:] + 1j*c2_Ml*dksi[:,1:]*dpsi[:,1:] - 1j*c3_Ml*ksi[:,1:]*psi[:,1:]
    dTj_Ml[:,:,0,1] = -1j*kj*d2ksi[:,1:]*ksi[:,1:] + 1j*c2_Ml*dksi[:,1:]**2 - 1j*c3_Ml*ksi[:,1:]**2
    dTj_Ml[:,:,1,0] = 1j*kj*d2psi[:,1:]*psi[:,1:] - 1j*c2_Ml*dpsi[:,1:]**2 + 1j*c3_Ml*psi[:,1:]**2
    dTj_Ml[:,:,1,1] = 1j*kj*d2psi[:,1:]*ksi[:,1:] - 1j*c2_Ml*dksi[:,1:]*dpsi[:,1:] + 1j*c3_Ml*ksi[:,1:]*psi[:,1:]
    
    if d_needle == 0 and d_needle == layer-1:
        dT_El = Tcu_El[:,:,d_needle,:,:]@dTj_El
        dT_Ml = Tcu_Ml[:,:,d_needle,:,:]@dTj_Ml
    elif d_needle == 0:
        dT_El = Tcu_El[:,:,d_needle,:,:]@dTj_El@Tcl_El[:,:,d_needle+1,:,:]
        dT_Ml = Tcu_Ml[:,:,d_needle,:,:]@dTj_Ml@Tcl_Ml[:,:,d_needle+1,:,:]
    elif d_needle == layer-1:
        dT_El = Tcu_El[:,:,d_needle,:,:]@dTj_El
        dT_Ml = Tcu_Ml[:,:,d_needle,:,:]@dTj_Ml
    else:
        dT_El = Tcu_El[:,:,d_needle,:,:]@dTj_El@Tcl_El[:,:,d_needle+1,:,:]
        dT_Ml = Tcu_Ml[:,:,d_needle,:,:]@dTj_Ml@Tcl_Ml[:,:,d_needle+1,:,:]
    
    dT11_El = dT_El[:,:,0,0]
    dT21_El = dT_El[:,:,1,0]
    dT11_Ml = dT_Ml[:,:,0,0]
    dT21_Ml = dT_Ml[:,:,1,0]

    dt_El = (1/T11_El**2)*(T11_El*dT21_El - T21_El*dT11_El)
    dt_Ml = (1/T11_Ml**2)*(T11_Ml*dT21_Ml - T21_Ml*dT11_Ml)

    # Efficiencies
    k_ext = np.real(n[:,0]*k)
    k_ext_temp = np.expand_dims(k_ext, axis=1) # wvl x lmax
    
    coeff_C = np.zeros((wvl, lmax))
    for n_l in range(1, lmax+1):
        coeff_C[:,n_l-1] = 2*n_l + 1
    
    dC_sca = 2*coeff_C*np.real(np.conj(t_El)*dt_El + np.conj(t_Ml)*dt_Ml)
    dC_abs = 4*coeff_C*np.real((1 + 2*np.conj(t_El))*dt_El + (1 + 2*np.conj(t_Ml))*dt_Ml)
    dC_ext = coeff_C*np.real(dt_El + dt_Ml)
    
    dC_sca_out = dC_sca*2*np.pi/k_ext_temp**2 # wvl x lmax
    dC_sca *= 2/(k_ext_temp**2*r[0]**2)
    dC_abs *= -1/(2*k_ext_temp**2*r[0]**2)
    dC_ext *= -2/(k_ext_temp**2*r[0]**2)
    
    dC_sca_out = np.sum(dC_sca_out, axis=1) # wvl
    dQ_sca = np.sum(dC_sca, axis=1)
    dQ_abs = np.sum(dC_abs, axis=1)
    dQ_ext = np.sum(dC_ext, axis=1)
    
    # Scattering Matrix
    coeff_S = np.zeros((wvl, th, lmax))
    for n_l in range(1, lmax+1):
        coeff_S[:,:,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))
    
    dt_El_temp = np.expand_dims(dt_El, axis=1)
    dt_Ml_temp = np.expand_dims(dt_Ml, axis=1)
    pi_l = np.expand_dims(pi_l, axis=0)
    tau_l = np.expand_dims(tau_l, axis=0)

    dS1 = coeff_S*(dt_El_temp*pi_l[:,:,1:] + dt_Ml_temp*tau_l[:,:,1:]) # wvl x th x lmax
    dS2 = coeff_S*(dt_El_temp*tau_l[:,:,1:] + dt_Ml_temp*pi_l[:,:,1:])
    
    dS1 = np.sum(dS1, axis=2) # wvl x th
    dS2 = np.sum(dS2, axis=2)
    
    # Angular Power Distribution
    k_ext_temp = np.expand_dims(np.expand_dims(k_ext, axis=1), axis=2) # wvl x th x ph
    S1_temp = np.expand_dims(S1, axis=2)
    S2_temp = np.expand_dims(S2, axis=2)
    dS1_temp = np.expand_dims(dS1, axis=2)
    dS2_temp = np.expand_dims(dS2, axis=2)
    
    sin_phi = np.sin(phi)
    sin_phi = np.expand_dims(np.expand_dims(sin_phi, axis=0), axis=0)
    cos_phi = np.cos(phi)
    cos_phi = np.expand_dims(np.expand_dims(cos_phi, axis=0), axis=0)
    
    Q_sca = np.expand_dims(np.expand_dims(Q_sca, axis=1), axis=2)
    dC_sca_out = np.expand_dims(np.expand_dims(dC_sca_out, axis=1), axis=2)
    
    d_diff_CS = (2/(k_ext_temp**2))*(np.real(np.conj(S1_temp)*dS1_temp)*sin_phi**2 + np.real(np.conj(S2_temp)*dS2_temp)*cos_phi**2) # wvl x th x ph
    dp = -(1/(k_ext_temp**2*(Q_sca*np.pi*r[0]**2)**2))*(np.abs(S1_temp)**2*sin_phi**2 + np.abs(S2_temp)**2*cos_phi**2)*dC_sca_out\
        + (2/(k_ext_temp**2*Q_sca*np.pi*r[0]**2))*(np.real(np.conj(S1_temp)*dS1_temp)*sin_phi**2 + np.real(np.conj(S2_temp)*dS2_temp)*cos_phi**2)
    
    return dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS

@jit(nopython=True, cache=True)
def build_topology_nucleation_derivative(Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, Q_sca, Q_abs, Q_ext, p,
                                         dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, lam, theta, phi):
    wvl = lam.size
    th = theta.size
    ph = phi.size    
    
    dMF_val = 0
            
    dMF_val += np.sum(2*(Q_sca - Q_sca_con[0,0,:])*Q_sca_con[0,1,:]**2*dQ_sca)
    dMF_val += np.sum(2*np.maximum(Q_sca - Q_sca_con[1,0,:], np.zeros(wvl))*Q_sca_con[1,1,:]**2*dQ_sca)
    dMF_val += np.sum(2*np.minimum(Q_sca - Q_sca_con[2,0,:], np.zeros(wvl))*Q_sca_con[2,1,:]**2*dQ_sca)
    
    dMF_val += np.sum(2*(Q_abs - Q_abs_con[0,0,:])*Q_abs_con[0,1,:]**2*dQ_abs)
    dMF_val += np.sum(2*np.maximum(Q_abs - Q_abs_con[1,0,:], np.zeros(wvl))*Q_abs_con[1,1,:]**2*dQ_abs)
    dMF_val += np.sum(2*np.minimum(Q_abs - Q_abs_con[2,0,:], np.zeros(wvl))*Q_abs_con[2,1,:]**2*dQ_abs)
    
    dMF_val += np.sum(2*(Q_ext - Q_ext_con[0,0,:])*Q_ext_con[0,1,:]**2*dQ_ext)
    dMF_val += np.sum(2*np.maximum(Q_ext - Q_ext_con[1,0,:], np.zeros(wvl))*Q_ext_con[1,1,:]**2*dQ_ext)
    dMF_val += np.sum(2*np.minimum(Q_ext - Q_ext_con[2,0,:], np.zeros(wvl))*Q_ext_con[2,1,:]**2*dQ_ext)
    
    dMF_val += np.sum(2*(p - p_con[0,0,:,:,:])*p_con[0,1,:,:,:]**2*dp)
    dMF_val += np.sum(2*np.maximum(p - p_con[1,0,:,:,:], np.zeros((wvl, th, ph)))*p_con[1,1,:,:,:]**2*dp)
    dMF_val += np.sum(2*np.minimum(p - p_con[2,0,:,:,:], np.zeros((wvl, th, ph)))*p_con[2,1,:,:,:]**2*dp)
    
    dMF_val += np.sum(diff_CS_con*d_diff_CS)
    
    return dMF_val

#@jit(nopython=True, cache=True)
def build_topology_nucleation_derivative_directional(Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, Q_sca, Q_abs, Q_ext, p,
                                                     dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, lam, theta, phi):
    th_Fwd = 11 # 5
    th0 = 291 # 355
    th1 = 302 # 360
    phi_tgt = np.array([0])
    wvl = 0
    
    dMF_val = 0
    denom = np.sum(p[wvl,th_Fwd:,:]*np.sin(theta[th_Fwd:])[np.newaxis,:,np.newaxis], axis=(1,2))
    d_denom = -np.sum(dp[wvl,th_Fwd:,:]*np.sin(theta[th_Fwd:])[np.newaxis,:,np.newaxis], axis=(1,2))/denom**2
    for i in range(phi_tgt.size):
        numer = np.sum(p[wvl,th0:th1,phi_tgt[i]]*np.sin(theta[th0:th1])[np.newaxis,:], axis=1)
        d_numer = np.sum(dp[wvl,th0:th1,phi_tgt[i]]*np.sin(theta[th0:th1])[np.newaxis,:], axis=1)
        
        dMF_val += -(d_numer/denom + numer*d_denom)
    
    return dMF_val