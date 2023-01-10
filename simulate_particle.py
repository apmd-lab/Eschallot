import numpy as np
import special_functions as spec
import tmm_mie as tmm

def tmm_mie_jit_wrapper(lam, theta, phi, pol, r, n, pool=None):
    k = (2*np.pi)/lam
    TE = np.cos(pol)**2
    TM = np.sin(pol)**2
    
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
    
    # Legendre polynomials
    pi_l = spec.pi_n(theta, lmax)
    tau_l = spec.tau_n(theta, lmax, pi_l)
    
    x = np.zeros((wvl, layer)).astype(np.complex128)
    x_tilde = np.zeros((wvl, layer)).astype(np.complex128)
    eta_tilde = np.zeros((wvl, layer)).astype(np.complex128)
    ksi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128) # First index: 0 --> no tilde, 1 --> tilde
    dksi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)
    psi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)
    dpsi = np.zeros((2, wvl, lmax+1, layer)).astype(np.complex128)

    for l in range(layer):
        x[:,l] = n[:,l+1]*k*r[l]
        eta_tilde[:,l] = n[:,l+1]/n[:,l]
        x_tilde[:,l] = x[:,l]/eta_tilde[:,l]

        ksi[0,:,:,l], dksi[0,:,:,l] = spec.RB_ksi(x[:,l], lmax)
        psi[0,:,:,l], dpsi[0,:,:,l] = spec.RB_psi(x[:,l], lmax)
        ksi[1,:,:,l], dksi[1,:,:,l] = spec.RB_ksi(x_tilde[:,l], lmax)
        psi[1,:,:,l], dpsi[1,:,:,l] = spec.RB_psi(x_tilde[:,l], lmax)
    
    Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
        Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam, theta, phi, lmax, k, TE, TM, r, n,
                                                                                psi, dpsi, ksi, dksi, pi_l, tau_l,
                                                                                eta_tilde)
    
    return Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM