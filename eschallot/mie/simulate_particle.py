import numpy as np
import eschallot.mie.special_functions as spec
import eschallot.mie.tmm_mie as tmm
import eschallot.util.read_mat_data as rmd

def simulate(lam, theta, phi, r, n):
    k = (2*np.pi)/lam
    
    wvl = np.size(lam)
    layer = np.size(r)
    
    x_max = np.max(n[:,0]*k*r[0])
    if x_max <= 8:
        nstop = np.round(x_max + 4*x_max**(1/3) + 1)
    elif x_max <= 4200:
        nstop = np.round(x_max + 4.05*x_max**(1/3) + 2)
    elif x_max <= 20000:
        nstop = np.round(x_max + 4*x_max**(1/3) + 2)
    else:
        raise ValueError('x_max too large')
    x1 = np.max(np.abs(n[:,1]*k*r[0]))
    if layer == 1:
        x2 = 0
    else:
        x2 = np.max(np.abs(n[:,1]*k*r[1]))
    lmax = int(np.real(np.round(np.max(np.array([nstop,x1,x2]))) + 15))
    
    while True:
        # Legendre polynomials
        pi_l = spec.pi_n(theta, lmax)
        tau_l = spec.tau_n(theta, lmax, pi_l)
        
        ksi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128) # First index: 0 --> no tilde, 1 --> tilde
        dksi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)
        psi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)
        dpsi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)
    
        k = k[:,np.newaxis]
        r = r[np.newaxis,:]
        
        x = n[:,1:]*k*r
        eta_tilde = n[:,1:]/n[:,:-1]
        x_tilde = x/eta_tilde
    
        ksi[0,:,:,:], dksi[0,:,:,:] = spec.RB_ksi(x, lmax)
        psi[0,:,:,:], dpsi[0,:,:,:] = spec.RB_psi(x, lmax)
        ksi[1,:,:,:], dksi[1,:,:,:] = spec.RB_ksi(x_tilde, lmax)
        psi[1,:,:,:], dpsi[1,:,:,:] = spec.RB_psi(x_tilde, lmax)
        
        k = k.reshape(-1)
        r = r.reshape(-1)
        
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml,\
            Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam, theta, phi, lmax, k, r, n,
                                                                                    psi, dpsi, ksi, dksi, pi_l, tau_l,
                                                                                    eta_tilde)
        
        nancheck = np.isnan(np.sum(t_El, axis=0))
        if np.sum(nancheck) > 0:
            lmax = np.min(np.argwhere(nancheck)) - 1
        else:
            break

    return Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM
