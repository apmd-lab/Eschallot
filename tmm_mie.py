import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def calculate_tmm(get_matrix, layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde):
    # Matrix for each layer
    T_El = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)
    T_Ml = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)
    
    # Cumulative matrix from outer to inner layers
    Tcu_El = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)
    Tcu_Ml = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)
    
    if get_matrix:
        # Cumulative matrix from inner to outer layers
        Tcl_El = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)
        Tcl_Ml = np.zeros((layer, 2, 2, wvl, lmax)).astype(np.complex128)

    # Calculate characteristic matrices for each layer
    for l in range(layer):
        for n_l in range(1, lmax+1):
            for w in range(wvl):
                Tj_El = np.zeros((2, 2)).astype(np.complex128)
                Tj_Ml = np.zeros((2, 2)).astype(np.complex128)
                
                Tj_El[0,0] = dksi[1,w,n_l,l]*psi[0,w,n_l,l] - ksi[1,w,n_l,l]*dpsi[0,w,n_l,l]/eta_tilde[w,l]
                Tj_El[0,1] = dksi[1,w,n_l,l]*ksi[0,w,n_l,l] - ksi[1,w,n_l,l]*dksi[0,w,n_l,l]/eta_tilde[w,l]
                Tj_El[1,0] = -dpsi[1,w,n_l,l]*psi[0,w,n_l,l] + psi[1,w,n_l,l]*dpsi[0,w,n_l,l]/eta_tilde[w,l]
                Tj_El[1,1] = -dpsi[1,w,n_l,l]*ksi[0,w,n_l,l] + psi[1,w,n_l,l]*dksi[0,w,n_l,l]/eta_tilde[w,l]
                Tj_El *= -1j
                
                Tj_Ml[0,0] = dksi[1,w,n_l,l]*psi[0,w,n_l,l]/eta_tilde[w,l] - ksi[1,w,n_l,l]*dpsi[0,w,n_l,l]
                Tj_Ml[0,1] = dksi[1,w,n_l,l]*ksi[0,w,n_l,l]/eta_tilde[w,l] - ksi[1,w,n_l,l]*dksi[0,w,n_l,l]
                Tj_Ml[1,0] = -dpsi[1,w,n_l,l]*psi[0,w,n_l,l]/eta_tilde[w,l] + psi[1,w,n_l,l]*dpsi[0,w,n_l,l]
                Tj_Ml[1,1] = -dpsi[1,w,n_l,l]*ksi[0,w,n_l,l]/eta_tilde[w,l] + psi[1,w,n_l,l]*dksi[0,w,n_l,l]
                Tj_Ml *= -1j
    
                T_El[l,:,:,w,n_l-1] = Tj_El
                T_Ml[l,:,:,w,n_l-1] = Tj_Ml
        
    # Calculate cumulative matrices
    Tcu_El[0,:,:,:,:] = T_El[0,:,:,:,:]
    Tcu_Ml[0,:,:,:,:] = T_Ml[0,:,:,:,:]
    if get_matrix:
        Tcl_El[-1,:,:,:,:] = T_El[-1,:,:,:,:]
        Tcl_Ml[-1,:,:,:,:] = T_Ml[-1,:,:,:,:]
    
    for l in range(1, layer):
        for n_l in range(lmax):
            for w in range(wvl):
                Tcu_El[l,:,:,w,n_l] = np.ascontiguousarray(Tcu_El[l-1,:,:,w,n_l])@np.ascontiguousarray(T_El[l,:,:,w,n_l])
                Tcu_Ml[l,:,:,w,n_l] = np.ascontiguousarray(Tcu_Ml[l-1,:,:,w,n_l])@np.ascontiguousarray(T_Ml[l,:,:,w,n_l])
    
    if get_matrix:
        for l in range(layer-2, -1, -1):
            for n_l in range(lmax):
                for w in range(wvl):
                    Tcl_El[l,:,:,w,n_l] = np.ascontiguousarray(T_El[l,:,:,w,n_l])@np.ascontiguousarray(Tcl_El[l+1,:,:,w,n_l])
                    Tcl_Ml[l,:,:,w,n_l] = np.ascontiguousarray(T_Ml[l,:,:,w,n_l])@np.ascontiguousarray(Tcl_Ml[l+1,:,:,w,n_l])
    
    # Only forward matrix is necessary (since the reverse can by found by symmetry)
    T11_El = Tcu_El[-1,0,0,:,:]
    T21_El = Tcu_El[-1,1,0,:,:]
    T11_Ml = Tcu_Ml[-1,0,0,:,:]
    T21_Ml = Tcu_Ml[-1,1,0,:,:]
    
    # Multipole expansion coefficients
    t_El = T21_El/T11_El
    t_Ml = T21_Ml/T11_Ml
    
    if get_matrix:
        Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml
    else:
        return t_El, t_Ml

@jit(nopython=True, cache=True)
def efficiencies(lam, theta, phi, lmax, k, ml_TE, ml_TM, r, n, psi, dpsi, ksi, dksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size
    ph = phi.size

    t_El, t_Ml = calculate_tmm(0, layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)
    
    # Cross sections
    k_ext = np.real(n[:,0]*k)
    C_sca_mpE = np.zeros((wvl, lmax)).astype(np.float64)
    C_sca_mpM = np.zeros((wvl, lmax)).astype(np.float64)
    C_sca = np.zeros((wvl, lmax)).astype(np.float64)
    C_abs = np.zeros((wvl, lmax)).astype(np.float64)
    C_ext = np.zeros((wvl, lmax)).astype(np.float64)
    for n_l in range(1, lmax+1):
        C_sca_mpE[:,n_l-1] = (2/(k_ext**2*r[0]**2))*(2*n_l + 1)*np.abs(t_El[:,n_l-1])**2
        C_sca_mpM[:,n_l-1] = (2/(k_ext**2*r[0]**2))*(2*n_l + 1)*np.abs(t_Ml[:,n_l-1])**2
        C_sca[:,n_l-1] = (2*n_l + 1)*(np.abs(t_El[:,n_l-1])**2 + np.abs(t_Ml[:,n_l-1])**2)
        C_abs[:,n_l-1] = (2*n_l + 1)*(2 - np.abs(1 + 2*t_El[:,n_l-1])**2 - np.abs(1 + 2*t_Ml[:,n_l-1])**2)
        C_ext[:,n_l-1] = (2*n_l + 1)*(np.real(t_El[:,n_l-1]) + np.real(t_Ml[:,n_l-1]))
    C_sca_out = np.real((2*np.pi/k_ext**2)*np.sum(C_sca, axis=1))
    Q_sca_mpE = np.real(C_sca_mpE)
    Q_sca_mpM = np.real(C_sca_mpM)
    Q_sca = np.real((2/(k_ext**2*r[0]**2))*np.sum(C_sca, axis=1))
    Q_abs = np.real((1/(2*k_ext**2*r[0]**2))*np.sum(C_abs, axis=1))
    Q_ext = np.real((-2/(k_ext**2*r[0]**2))*np.sum(C_ext, axis=1))
    
    # Scattering amplitudes
    S1 = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S2 = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S1_mpE = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S1_mpM = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S2_mpE = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S2_mpM = np.zeros((wvl, th, lmax)).astype(np.complex128)
    for t in range(th):
        for n_l in range(1, lmax+1):
            S1[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*(t_El[:,n_l-1]*pi_l[t,n_l] + t_Ml[:,n_l-1]*tau_l[t,n_l])
            S2[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*(t_El[:,n_l-1]*tau_l[t,n_l] + t_Ml[:,n_l-1]*pi_l[t,n_l])
            S1_mpE[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*t_El[:,n_l-1]*pi_l[t,n_l]
            S1_mpM[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*t_Ml[:,n_l-1]*tau_l[t,n_l]
            S2_mpE[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*t_El[:,n_l-1]*tau_l[t,n_l]
            S2_mpM[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*t_Ml[:,n_l-1]*pi_l[t,n_l]
    S1 = np.sum(S1, axis=2)
    S2 = np.sum(S2, axis=2)
    
    # Scattering phase function
    p = np.zeros((wvl, th, ph)).astype(np.float64)
    for t in range(th):
        for n_p in range(ph):
            p[:,t,n_p] = ml_TE*np.abs(S1[:,t])**2*np.sin(phi[n_p])**2 + ml_TM*np.abs(S2[:,t])**2*np.cos(phi[n_p])**2
            p[:,t,n_p] /= k_ext**2*C_sca_out

    return Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM

@jit(nopython=True, cache=True)
def transfer_matrix(lam, theta, phi, lmax, k, ml_TE, ml_TM, r, n, psi, dpsi, ksi, dksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size

    Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml,\
        T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml = calculate_tmm(1, layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)
    
    # Scattering amplitudes
    S1 = np.zeros((wvl, th, lmax)).astype(np.complex128)
    S2 = np.zeros((wvl, th, lmax)).astype(np.complex128)
    for t in range(th):
        for n_l in range(1, lmax+1):
            S1[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*(t_El[:,n_l-1]*pi_l[t,n_l] + t_Ml[:,n_l-1]*tau_l[t,n_l])
            S2[:,t,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))*(t_El[:,n_l-1]*tau_l[t,n_l] + t_Ml[:,n_l-1]*pi_l[t,n_l])
    S1 = np.sum(S1, axis=2)
    S2 = np.sum(S2, axis=2)
    
    return Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2