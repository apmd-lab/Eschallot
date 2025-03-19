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

def efficiency_shape_derivatives(lam, theta, phi, lmax, k, r, n, psi, dpsi, d2psi, ksi, dksi, d2ksi, pi_l, tau_l, eta_tilde):
    wvl = lam.size
    layer = r.size
    th = theta.size
    ph = phi.size
    
    Tcu_El, Tcu_Ml, Tcl_El, Tcl_Ml,\
        T11_El, T21_El, T11_Ml, T21_Ml, t_El, t_Ml = calculate_tmm_matrix(layer, wvl, lmax, psi, dpsi, ksi, dksi, eta_tilde)

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
    
    return Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml

def efficiency_topology_derivatives(r_needle, d_needle, n_needle, Q_sca, Q_abs, Q_ext, p, diff_CS,
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
    
    dt_El = np.expand_dims(dt_El, axis=-1)
    dt_Ml = np.expand_dims(dt_El, axis=-1)
    dQ_sca = np.expand_dims(dQ_sca, axis=-1)
    dQ_abs = np.expand_dims(dQ_abs, axis=-1)
    dQ_ext = np.expand_dims(dQ_ext, axis=-1)
    dp = np.expand_dims(dp, axis=-1)
    d_diff_CS = np.expand_dims(d_diff_CS, axis=-1)
    
    return dt_El, dt_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS