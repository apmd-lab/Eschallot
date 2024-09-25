import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def calculate_tmm_matrix(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde):
    # Matrix for each layer
    T_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    T_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    
    # Cumulative matrix from outer to inner layers
    Tcu_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    Tcu_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    
    # Cumulative matrix from inner to outer layers
    Tcl_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    Tcl_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)

    # Calculate characteristic matrices for each layer
    eta_tilde_temp = np.expand_dims(eta_tilde, axis=1)
                
    T_El[:,:,:,0,0] = dksi[1,:,1:,:]*psi[0,:,1:,:] - ksi[1,:,1:,:]*dpsi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,0,1] = dksi[1,:,1:,:]*ksi[0,:,1:,:] - ksi[1,:,1:,:]*dksi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,1,0] = -dpsi[1,:,1:,:]*psi[0,:,1:,:] + psi[1,:,1:,:]*dpsi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,1,1] = -dpsi[1,:,1:,:]*ksi[0,:,1:,:] + psi[1,:,1:,:]*dksi[0,:,1:,:]/eta_tilde_temp
    T_El *= -1j
    
    T_Ml[:,:,:,0,0] = dksi[1,:,1:,:]*psi[0,:,1:,:]/eta_tilde_temp - ksi[1,:,1:,:]*dpsi[0,:,1:,:]
    T_Ml[:,:,:,0,1] = dksi[1,:,1:,:]*ksi[0,:,1:,:]/eta_tilde_temp - ksi[1,:,1:,:]*dksi[0,:,1:,:]
    T_Ml[:,:,:,1,0] = -dpsi[1,:,1:,:]*psi[0,:,1:,:]/eta_tilde_temp + psi[1,:,1:,:]*dpsi[0,:,1:,:]
    T_Ml[:,:,:,1,1] = -dpsi[1,:,1:,:]*ksi[0,:,1:,:]/eta_tilde_temp + psi[1,:,1:,:]*dksi[0,:,1:,:]
    T_Ml *= -1j
        
    # Calculate cumulative matrices
    Tcu_El[:,:,0,:,:] = T_El[:,:,0,:,:]
    Tcu_Ml[:,:,0,:,:] = T_Ml[:,:,0,:,:]
    Tcl_El[:,:,-1,:,:] = T_El[:,:,-1,:,:]
    Tcl_Ml[:,:,-1,:,:] = T_Ml[:,:,-1,:,:]
    
    for l in range(1, layer):
        for n_l in range(lmax):
            for w in range(wvl):
                Tcu_El[w,n_l,l,:,:] = np.ascontiguousarray(Tcu_El[w,n_l,l-1,:,:])@np.ascontiguousarray(T_El[w,n_l,l,:,:])
                Tcu_Ml[w,n_l,l,:,:] = np.ascontiguousarray(Tcu_Ml[w,n_l,l-1,:,:])@np.ascontiguousarray(T_Ml[w,n_l,l,:,:])
    
    for l in range(layer-2, -1, -1):
        for n_l in range(lmax):
            for w in range(wvl):
                Tcl_El[w,n_l,l,:,:] = np.ascontiguousarray(T_El[w,n_l,l,:,:])@np.ascontiguousarray(Tcl_El[w,n_l,l+1,:,:])
                Tcl_Ml[w,n_l,l,:,:] = np.ascontiguousarray(T_Ml[w,n_l,l,:,:])@np.ascontiguousarray(Tcl_Ml[w,n_l,l+1,:,:])
    
    # Only forward matrix is necessary (since the reverse can by found by symmetry)
    T11_El = Tcu_El[:,:,-1,0,0]
    T21_El = Tcu_El[:,:,-1,1,0]
    T11_Ml = Tcu_Ml[:,:,-1,0,0]
    T21_Ml = Tcu_Ml[:,:,-1,1,0]
    
    # Multipole expansion coefficients
    t_El = T21_El/T11_El
    t_Ml = T21_Ml/T11_Ml
    
    return Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml

@jit(nopython=True, cache=True)
def calculate_tmm_coeff(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde):
    # Matrix for each layer
    T_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    T_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    
    # Cumulative matrix from outer to inner layers
    Tcu_El = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)
    Tcu_Ml = np.zeros((wvl, lmax, layer, 2, 2)).astype(np.complex128)

    # Calculate characteristic matrices for each layer
    eta_tilde_temp = np.expand_dims(eta_tilde, axis=1)
                
    T_El[:,:,:,0,0] = dksi[1,:,1:,:]*psi[0,:,1:,:] - ksi[1,:,1:,:]*dpsi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,0,1] = dksi[1,:,1:,:]*ksi[0,:,1:,:] - ksi[1,:,1:,:]*dksi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,1,0] = -dpsi[1,:,1:,:]*psi[0,:,1:,:] + psi[1,:,1:,:]*dpsi[0,:,1:,:]/eta_tilde_temp
    T_El[:,:,:,1,1] = -dpsi[1,:,1:,:]*ksi[0,:,1:,:] + psi[1,:,1:,:]*dksi[0,:,1:,:]/eta_tilde_temp
    T_El *= -1j
    
    T_Ml[:,:,:,0,0] = dksi[1,:,1:,:]*psi[0,:,1:,:]/eta_tilde_temp - ksi[1,:,1:,:]*dpsi[0,:,1:,:]
    T_Ml[:,:,:,0,1] = dksi[1,:,1:,:]*ksi[0,:,1:,:]/eta_tilde_temp - ksi[1,:,1:,:]*dksi[0,:,1:,:]
    T_Ml[:,:,:,1,0] = -dpsi[1,:,1:,:]*psi[0,:,1:,:]/eta_tilde_temp + psi[1,:,1:,:]*dpsi[0,:,1:,:]
    T_Ml[:,:,:,1,1] = -dpsi[1,:,1:,:]*ksi[0,:,1:,:]/eta_tilde_temp + psi[1,:,1:,:]*dksi[0,:,1:,:]
    T_Ml *= -1j
        
    # Calculate cumulative matrices
    Tcu_El[:,:,0,:,:] = T_El[:,:,0,:,:]
    Tcu_Ml[:,:,0,:,:] = T_Ml[:,:,0,:,:]
    
    for l in range(1, layer):
        for n_l in range(lmax):
            for w in range(wvl):
                Tcu_El[w,n_l,l,:,:] = np.ascontiguousarray(Tcu_El[w,n_l,l-1,:,:])@np.ascontiguousarray(T_El[w,n_l,l,:,:])
                Tcu_Ml[w,n_l,l,:,:] = np.ascontiguousarray(Tcu_Ml[w,n_l,l-1,:,:])@np.ascontiguousarray(T_Ml[w,n_l,l,:,:])

    # Only forward matrix is necessary (since the reverse can by found by symmetry)
    T11_El = Tcu_El[:,:,-1,0,0]
    T21_El = Tcu_El[:,:,-1,1,0]
    T11_Ml = Tcu_Ml[:,:,-1,0,0]
    T21_Ml = Tcu_Ml[:,:,-1,1,0]
    
    # Multipole expansion coefficients
    t_El = T21_El/T11_El
    t_Ml = T21_Ml/T11_Ml
    
    return t_El, t_Ml

@jit(nopython=True, cache=True, debug=True)
def efficiencies(lam, theta, phi, lmax, k, r, n, psi, dpsi, ksi, dksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size
    ph = phi.size

    t_El, t_Ml = calculate_tmm_coeff(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)
    
    # Cross sections
    k_ext = np.real(n[:,0]*k) # wvl
    k_ext_temp = np.expand_dims(k_ext, axis=1)
    coeff_C = np.zeros((wvl, lmax))
    for n_l in range(1, lmax+1):
        coeff_C[:,n_l-1] = (2*n_l + 1)
    
    C_sca_mpE = coeff_C*np.abs(t_El)**2 # wvl x lmax
    C_sca_mpM = coeff_C*np.abs(t_Ml)**2
    C_sca = C_sca_mpE + C_sca_mpM
    C_abs = coeff_C*(2 - np.abs(1 + 2*t_El)**2 - np.abs(1 + 2*t_Ml)**2)
    C_ext = coeff_C*(np.real(t_El) + np.real(t_Ml))

    C_sca_mpE *= (2/(k_ext_temp**2*r[0]**2))
    C_sca_mpM *= (2/(k_ext_temp**2*r[0]**2))
    C_sca = np.real((2*np.pi/k_ext**2)*np.sum(C_sca, axis=1)) # wvl
    C_abs = np.real((np.pi/(2*k_ext**2))*np.sum(C_abs, axis=1))
    C_ext = np.real((-2*np.pi/k_ext**2)*np.sum(C_ext, axis=1))
    
    Q_sca_mpE = np.real(C_sca_mpE)
    Q_sca_mpM = np.real(C_sca_mpM)
    Q_sca = C_sca/(np.pi*r[0]**2)
    Q_abs = C_abs/(np.pi*r[0]**2)
    Q_ext = C_ext/(np.pi*r[0]**2)
    
    # t_El[:,0] = -0.5*np.cos(89*np.pi/180) + 0.5*np.sin(89*np.pi/180)
    # t_El[:,1] = -0.5*np.cos(89*np.pi/180) - 0.5*np.sin(89*np.pi/180)
    # t_Ml[:,:2] = -0.5*np.cos(89*np.pi/180) - 0.5*np.sin(89*np.pi/180)
    # t_El[:,:2] = -1
    # t_Ml[:,:2] = -1
    
    # Scattering amplitudes
    t_El_temp = np.expand_dims(t_El, axis=1)
    t_Ml_temp = np.expand_dims(t_Ml, axis=1)
    pi_l = np.expand_dims(pi_l, axis=0) # wvl x th x lmax
    tau_l = np.expand_dims(tau_l, axis=0)
    
    coeff_S = np.zeros((wvl, th, lmax))
    for n_l in range(1, lmax+1):
        coeff_S[:,:,n_l-1] = -((2*n_l + 1)/(n_l*(n_l + 1)))

    S1_mpE = coeff_S*t_El_temp*pi_l[:,:,1:]
    S1_mpM = coeff_S*t_Ml_temp*tau_l[:,:,1:]
    S1 = S1_mpE + S1_mpM # wvl x th x lmax
    
    S2_mpE = coeff_S*t_El_temp*tau_l[:,:,1:]
    S2_mpM = coeff_S*t_Ml_temp*pi_l[:,:,1:]
    S2 = S2_mpE + S2_mpM
    
    S1 = np.sum(S1, axis=2) # wvl x th
    S2 = np.sum(S2, axis=2)

    # S1 = S1[:,:,0]
    # S2 = S2[:,:,0]
    # S1 = S1_mpM[:,:,1]
    # S2 = S2_mpM[:,:,1]
    
    # Scattering angular distribution
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

    return Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM

@jit(nopython=True, cache=True)
def transfer_matrix(lam, theta, phi, lmax, k, r, n, psi, dpsi, ksi, dksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size

    Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml,\
        T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml = calculate_tmm_matrix(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)
    
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
    
    return Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml, T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml, S1, S2
