import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def RB_ksi(x, lmax):
    wvl = x.shape[0]
    layer = x.shape[1]

    a = x.real
    b = x.imag
    log_ksi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    log_ksi[:,0] = 1j*(a-np.pi/2) - b
    ksi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    log_dksi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    dksi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)

    d3n = D3n(x, lmax)
    for n_ind in range(1, lmax+1):
        log_ksi[:,n_ind,:] = log_ksi[:,n_ind-1,:] + np.log(n_ind/x - d3n[:,n_ind-1,:])
        log_dksi[:,n_ind,:] = log_ksi[:,n_ind,:] + np.log(d3n[:,n_ind,:])
    
    ksi = np.exp(log_ksi)
    dksi = np.exp(log_dksi)
    
    return ksi, dksi

@jit(nopython=True, cache=True)
def RB_psi(x, lmax):
    wvl = x.shape[0]
    layer = x.shape[1]

    a = x.real
    b = x.imag
    log_psi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    log_psi[:,0] = b - np.log(2) + np.log((1+np.exp(-2*b))*np.sin(a)+1j*(1-np.exp(-2*b))*np.cos(a))
    psi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    log_dpsi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    dpsi = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    
    d1n = D1n(x, lmax)
    for n_ind in range(1, lmax+1):
        log_psi[:,n_ind,:] = log_psi[:,n_ind-1,:] + np.log(n_ind/x - d1n[:,n_ind-1,:])
        log_dpsi[:,n_ind,:] = log_psi[:,n_ind,:] + np.log(d1n[:,n_ind,:])
    
    psi = np.exp(log_psi)
    dpsi = np.exp(log_dpsi)
    
    return psi, dpsi

@jit(nopython=True, cache=True)
def D1n(x, lmax):
    wvl = x.shape[0]
    layer = x.shape[1]

    d1n = np.zeros((wvl, lmax+1, layer)).astype(np.complex128)
    v = lmax + 1/2
    a1 = 2*v/x
    a2 = np.ones((wvl, layer)).astype(np.complex128)
    jn_ratio = np.ones((wvl, lmax+1, layer)).astype(np.complex128)
    
    for l in range(layer):
        for w in range(wvl):
            n = 1
            while a1[w,l] != a2[w,l] and n <= 1000:
                jn_ratio[w,lmax,l] *= a1[w,l]/a2[w,l]
                if n == 1:
                    a2[w,l] = -2*(v+1)/x[w,l]
                else:
                    a2[w,l] = 1/a2[w,l] + (-1)**(n+2)*2*(v+n)/x[w,l]
                a1[w,l] = 1/a1[w,l] + (-1)**(n+2)*2*(v+n)/x[w,l]
                n += 1
    
    for n_ind in range(lmax-1,-1,-1):
        jn_ratio[:,n_ind,:] = (2*n_ind+1)/x - 1/jn_ratio[:,n_ind+1,:]
        d1n[:,n_ind,:] = -n_ind/x + jn_ratio[:,n_ind,:]
    d1n[:,lmax,:] = -lmax/x + jn_ratio[:,lmax,:]

    return d1n

@jit(nopython=True, cache=True)
def D3n(x, lmax):
    wvl = x.shape[0]
    layer = x.shape[1]

    d3n = 1j*np.ones((wvl, lmax+1, layer)).astype(np.complex128)
    for n_ind in range(1, lmax+1):
        d3n[:,n_ind] = -n_ind/x + 1/(n_ind/x-d3n[:,n_ind-1])

    return d3n

@jit(nopython=True, cache=True)
def pi_n(theta, lmax):
    th = theta.size
    pi_array = np.zeros((th, lmax+1))
    pi_array[:,1] = 1
    
    for n_ind in range(2, lmax+1):
        pi_array[:,n_ind] = ((2*n_ind-1)/(n_ind-1))*np.cos(theta)*pi_array[:,n_ind-1] - (n_ind/(n_ind-1))*pi_array[:,n_ind-2]
    
    pi_array_fn = np.zeros((th, lmax+1))
    for n_ind in range(lmax+1):
        pi_array_fn[:,n_ind] = pi_array[:,n_ind]
    return pi_array_fn

@jit(nopython=True, cache=True)
def tau_n(theta, lmax, pi_array):
    th = theta.size
    tau_array = np.zeros((th, lmax+1))
    
    for n_ind in range(1, lmax+1):
        tau_array[:,n_ind] = n_ind*np.cos(theta)*pi_array[:,n_ind] - (n_ind+1)*pi_array[:,n_ind-1]
        
    tau_array_fn = np.zeros((th, lmax+1))
    for n_ind in range(lmax+1):
        tau_array_fn[:,n_ind] = tau_array[:,n_ind]
    return tau_array_fn
