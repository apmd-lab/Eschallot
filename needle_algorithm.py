import numpy as np
import special_functions as spec
import tmm_mie as tmm
import tmm_mie_derivatives as deriv_tmm
import matplotlib.pyplot as plt
import os
directory = os.path.dirname(os.path.realpath(__file__))
from scipy.optimize import minimize_scalar, minimize, LinearConstraint

class multilayer:
    def __init__(self, r, n, ban_needle, lam, theta, phi, pol):
        """ n: refractive index, dim --> layer(outer to inner) x lam(short to long)
        r: interface radius, dim --> layer(outer to inner) x 1 (excluding external medium)
        lam: wavelength, dim --> 1 x lam(short to long)
        theta: incident angle, dim --> 1 x 1
        pol: polarization angle, TE: 0 rad. dim --> 1 x 1 """
        
        # User-provided multilayer quantities
        self.n = n
        self.ban_needle = ban_needle
        self.lam = lam
        self.theta = theta
        self.phi = phi
        self.pol = pol

        # Derived quantities
        self.k = (2*np.pi)/lam
        self.TE = np.cos(pol)**2
        self.TM = np.sin(pol)**2

    def update(self, r, n, lmax=None):
        # Get Number of Orders (l)
        wvl = np.size(self.lam)
        layer = np.size(r)
        
        if lmax is None:
            x_max = np.max(n[:,0]*self.k*r[0])
            if x_max <= 8:
                nstop = np.round(x_max + 4*x_max**(1/3) + 1)
            elif x_max <= 4200:
                nstop = np.round(x_max + 4.05*x_max**(1/3) + 2)
            elif x_max <= 20000:
                nstop = np.round(x_max + 4*x_max**(1/3) + 2)
            else:
                raise ValueError('x_max too large')
                
            x1 = np.max(np.abs(n[:,1]*self.k*r[0]))
            if layer == 1:
                x2 = 0
            else:
                x2 = np.max(np.abs(n[:,1]*self.k*r[1]))
            # self.lmax = int(np.min((int(np.real(np.round(np.max(np.array([nstop,x1,x2]))) + 15)), 100)))
            self.lmax = int(np.real(np.round(np.max(np.array([nstop,x1,x2]))) + 15))
            # print('Number of Orders (l): ' + str(int(self.lmax)))
        else:
            self.lmax = lmax
        
        # Legendre polynomials
        self.pi_l = spec.pi_n(self.theta, self.lmax)
        self.tau_l = spec.tau_n(self.theta, self.lmax, self.pi_l)
        
        self.x = np.zeros((wvl, layer)).astype(np.complex128)
        self.x_tilde = np.zeros((wvl, layer)).astype(np.complex128)
        self.eta_tilde = np.zeros((wvl, layer)).astype(np.complex128)
        self.ksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128) # First index: 0 --> no tilde, 1 --> tilde
        self.dksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.d2ksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.psi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.dpsi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.d2psi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)

        for l in range(layer):
            self.x[:,l] = n[:,l+1]*self.k*r[l]
            self.eta_tilde[:,l] = n[:,l+1]/n[:,l]
            self.x_tilde[:,l] = self.x[:,l]/self.eta_tilde[:,l]

            self.ksi[0,:,:,l], self.dksi[0,:,:,l] = spec.RB_ksi(self.x[:,l], self.lmax)
            self.psi[0,:,:,l], self.dpsi[0,:,:,l] = spec.RB_psi(self.x[:,l], self.lmax)
            self.ksi[1,:,:,l], self.dksi[1,:,:,l] = spec.RB_ksi(self.x_tilde[:,l], self.lmax)
            self.psi[1,:,:,l], self.dpsi[1,:,:,l] = spec.RB_psi(self.x_tilde[:,l], self.lmax)
    
            for n_l in range(self.lmax+1):
                self.d2ksi[0,:,n_l,l] = -(1 - (n_l*(n_l + 1))/self.x[:,l]**2)*self.ksi[0,:,n_l,l]
                self.d2psi[0,:,n_l,l] = -(1 - (n_l*(n_l + 1))/self.x[:,l]**2)*self.psi[0,:,n_l,l]
                self.d2ksi[1,:,n_l,l] = -(1 - (n_l*(n_l + 1))/self.x_tilde[:,l]**2)*self.ksi[1,:,n_l,l]
                self.d2psi[1,:,n_l,l] = -(1 - (n_l*(n_l + 1))/self.x_tilde[:,l]**2)*self.psi[1,:,n_l,l]

def jacobian(r, n, iteration, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con, ml_init):
    Jac = np.zeros(((3+ml_init.theta.size*ml_init.phi.size)*lam.size, np.size(r)))
    ml_init.update(r, n)
    Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
        dQ_sca, dQ_abs, dQ_ext, dp, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                                                    ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi, ml_init.d2psi,
                                                                                    ml_init.ksi, ml_init.dksi, ml_init.d2ksi, ml_init.pi_l, ml_init.tau_l,
                                                                                    ml_init.eta_tilde)
    
    nancheck = np.isnan(np.sum(t_El, axis=0)) + np.isnan(np.sum(dt_El, axis=(0,2)))
    lmax = None
    if np.sum(nancheck) > 0:
        lmax = np.min(np.argwhere(nancheck)) - 1
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                                                        ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi, ml_init.d2psi,
                                                                                        ml_init.ksi, ml_init.dksi, ml_init.d2ksi, ml_init.pi_l, ml_init.tau_l,
                                                                                        ml_init.eta_tilde)
    
    if np.sum(np.isnan(dQ_sca)) + np.sum(np.isnan(dQ_abs)) + np.sum(np.isnan(dQ_ext)) + np.sum(np.isnan(dp)) > 0:
        r[-1] = np.max((0, r[-1]))
        for l in range(r.size-2, -1, -1):
            if r[l] < r[l+1]:
                r[l] = r[l+1]
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml,\
            dQ_sca, dQ_abs, dQ_ext, dp, dt_El, dt_Ml = deriv_tmm.efficiency_derivatives(lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                                                        ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi, ml_init.d2psi,
                                                                                        ml_init.ksi, ml_init.dksi, ml_init.d2ksi, ml_init.pi_l, ml_init.tau_l,
                                                                                        ml_init.eta_tilde)
        
    for l in range(r.size):
        cnt = 0
        for w in range(lam.size):
            if not np.isnan(Q_sca_con[0,0,w]):
                Jac[cnt,l] = 2*(Q_sca[w] - Q_sca_con[0,0,w])*dQ_sca[w,l]*Q_sca_con[0,1,w]**2
                cnt += 1
            if not np.isnan(Q_sca_con[1,0,w]):
                Jac[cnt,l] = 2*np.max((Q_sca[w] - Q_sca_con[1,0,w], 0))*dQ_sca[w,l]*Q_sca_con[1,1,w]**2
                cnt += 1
            if not np.isnan(Q_sca_con[2,0,w]):
                Jac[cnt,l] = 2*np.min((Q_sca[w] - Q_sca_con[2,0,w], 0))*dQ_sca[w,l]*Q_sca_con[2,1,w]**2
                cnt += 1
        for w in range(lam.size):
            if not np.isnan(Q_abs_con[0,0,w]):
                Jac[cnt,l] = 2*(Q_abs[w] - Q_abs_con[0,0,w])*dQ_abs[w,l]*Q_abs_con[0,1,w]**2
                cnt += 1
            if not np.isnan(Q_abs_con[1,0,w]):
                Jac[cnt,l] = 2*np.max((Q_abs[w] - Q_abs_con[1,0,w], 0))*dQ_abs[w,l]*Q_abs_con[1,1,w]**2
                cnt += 1
            if not np.isnan(Q_abs_con[2,0,w]):
                Jac[cnt,l] = 2*np.min((Q_abs[w] - Q_abs_con[2,0,w], 0))*dQ_abs[w,l]*Q_abs_con[2,1,w]**2
                cnt += 1
        for w in range(lam.size):
            if not np.isnan(Q_ext_con[0,0,w]):
                Jac[cnt,l] = 2*(Q_ext[w] - Q_ext_con[0,0,w])*dQ_ext[w,l]*Q_ext_con[0,1,w]**2
                cnt += 1
            if not np.isnan(Q_ext_con[1,0,w]):
                Jac[cnt,l] = 2*np.max((Q_ext[w] - Q_ext_con[1,0,w], 0))*dQ_ext[w,l]*Q_ext_con[1,1,w]**2
                cnt += 1
            if not np.isnan(Q_ext_con[2,0,w]):
                Jac[cnt,l] = 2*np.min((Q_ext[w] - Q_ext_con[2,0,w], 0))*dQ_ext[w,l]*Q_ext_con[2,1,w]**2
                cnt += 1
        for w in range(lam.size):
            for t in range(ml_init.theta.size):
                for n_p in range(ml_init.phi.size):
                    if not np.isnan(p_con[0,0,w,t,n_p]):
                        Jac[cnt,l] = 2*(p[w,t,n_p] - p_con[0,0,w,t,n_p])*dp[w,t,n_p,l]*p_con[0,1,w,t,n_p]**2
                        cnt += 1
                    if not np.isnan(p_con[1,0,w,t,n_p]):
                        Jac[cnt,l] = 2*np.max((p[w,t,n_p] - p_con[1,0,w,t,n_p], 0))*dp[w,t,n_p,l]*p_con[1,1,w,t,n_p]**2
                        cnt += 1
                    if not np.isnan(p_con[2,0,w,t,n_p]):
                        Jac[cnt,l] = 2*np.min((p[w,t,n_p] - p_con[2,0,w,t,n_p], 0))*dp[w,t,n_p,l]*p_con[2,1,w,t,n_p]**2
                        cnt += 1

    return np.sum(Jac[:cnt,:], axis=0)

def merit_fct(r, n, iteration, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con, ml_init):
    """ Constraints must be in the form (2 x ...) where 0: target value, 1: weight
    """
    ml_init.update(r, n)
    Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
        S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                          ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi,
                                                          ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                          ml_init.eta_tilde)
    
    nancheck = np.isnan(np.sum(t_El, axis=0))
    if np.sum(nancheck) > 0:
        lmax = np.min(np.argwhere(nancheck)) - 1
        ml_init.update(r, n, lmax=lmax)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                              ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi,
                                                              ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                              ml_init.eta_tilde)
    
    res = np.zeros((3+ml_init.theta.size*ml_init.phi.size)*lam.size)
    cnt = 0
    for w in range(lam.size):
        if not np.isnan(Q_sca_con[0,0,w]):
            res[cnt] = (Q_sca[w] - Q_sca_con[0,0,w])*Q_sca_con[0,1,w]
            cnt += 1
        if not np.isnan(Q_sca_con[1,0,w]):
            res[cnt] = np.max((Q_sca[w] - Q_sca_con[1,0,w], 0))*Q_sca_con[1,1,w]
            cnt += 1
        if not np.isnan(Q_sca_con[2,0,w]):
            res[cnt] = np.min((Q_sca[w] - Q_sca_con[2,0,w], 0))*Q_sca_con[2,1,w]
            cnt += 1
    for w in range(lam.size):
        if not np.isnan(Q_abs_con[0,0,w]):
            res[cnt] = (Q_abs[w] - Q_abs_con[0,0,w])*Q_abs_con[0,1,w]
            cnt += 1
        if not np.isnan(Q_abs_con[1,0,w]):
            res[cnt] = np.max((Q_abs[w] - Q_abs_con[1,0,w], 0))*Q_abs_con[1,1,w]
            cnt += 1
        if not np.isnan(Q_abs_con[2,0,w]):
            res[cnt] = np.min((Q_abs[w] - Q_abs_con[2,0,w], 0))*Q_abs_con[2,1,w]
            cnt += 1
    for w in range(lam.size):
        if not np.isnan(Q_ext_con[0,0,w]):
            res[cnt] = (Q_ext[w] - Q_ext_con[0,0,w])*Q_ext_con[0,1,w]
            cnt += 1
        if not np.isnan(Q_ext_con[1,0,w]):
            res[cnt] = np.max((Q_ext[w] - Q_ext_con[1,0,w], 0))*Q_ext_con[1,1,w]
            cnt += 1
        if not np.isnan(Q_ext_con[2,0,w]):
            res[cnt] = np.min((Q_ext[w] - Q_ext_con[2,0,w], 0))*Q_ext_con[2,1,w]
            cnt += 1
    for w in range(lam.size):
        for t in range(ml_init.theta.size):
            for n_p in range(ml_init.phi.size):
                if not np.isnan(p_con[0,0,w,t,n_p]):
                    res[cnt] = (p[w,t,n_p] - p_con[0,0,w,t,n_p])*p_con[0,1,w,t,n_p]
                    cnt += 1
                if not np.isnan(p_con[1,0,w,t,n_p]):
                    res[cnt] = np.max((p[w,t,n_p] - p_con[1,0,w,t,n_p], 0))*p_con[1,1,w,t,n_p]
                    cnt += 1
                if not np.isnan(p_con[2,0,w,t,n_p]):
                    res[cnt] = np.min((p[w,t,n_p] - p_con[2,0,w,t,n_p], 0))*p_con[2,1,w,t,n_p]
                    cnt += 1
    
    return np.sum(res[:cnt]**2)

def refine_r(iteration, ml, r0, n, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con, d_low):
    A = np.zeros((r0.size, r0.size))
    for l in range(r0.size):
        A[l,l] = 1
        if l < r0.size - 1:
            A[l,l+1] = -1
    constr = LinearConstraint(A, lb=np.ones(r0.size), ub=np.inf*np.ones(r0.size))
    scale = 1
    while True:
        result = minimize(merit_fct, r0, args=(n, iteration, lam, Q_sca_con*scale, Q_abs_con*scale, Q_ext_con*scale, p_con*scale, ml),
                          method='trust-constr', jac=jacobian, constraints=constr, options={'verbose': 2, 'gtol': 1e-4, 'xtol': 1e-4,
                                                                                            'maxiter': 200})

        if result.status == 0 and scale < 10:
            scale *= 1.5
        else:
            break
    
    Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
        S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam, ml.theta, ml.phi, ml.lmax, ml.k,
                                                          ml.TE, ml.TM, result.x, n, ml.psi, ml.dpsi,
                                                          ml.ksi, ml.dksi, ml.pi_l, ml.tau_l,
                                                          ml.eta_tilde)
    
    plot_spectra(iteration, lam, ml.theta, ml.phi, Q_sca, Q_abs, Q_ext, p, Q_sca_con, Q_abs_con, Q_ext_con, p_con, result.x, n)
    
    return result.x, result.fun/scale**2, Q_sca, Q_abs, Q_ext, p

def init_needle(ml, r, n, lam):
    # make original object for reference in grad_needle
    result = tmm.transfer_matrix(lam, ml.theta, ml.phi, ml.lmax, ml.k,
                                 ml.TE, ml.TM, r, n, ml.psi, ml.dpsi,
                                 ml.ksi, ml.dksi, ml.pi_l, ml.tau_l,
                                 ml.eta_tilde)
    ml.Tcu_El = result[0]
    ml.Tcu_Ml = result[1]
    ml.Tcl_El = result[2]
    ml.Tcl_Ml = result[3]
    ml.T11_El = result[4]
    ml.T21_El = result[5]
    ml.T11_Ml = result[6]
    ml.T21_Ml = result[7]
    ml.t_El = result[8]
    ml.t_Ml = result[9]
    ml.S1 = result[10]
    ml.S2 = result[11]

def insert_needle(ml_init, mat_dict, mat_needle, r, n, ban_needle, lam, Q_sca, Q_abs, Q_ext, p,
                  Q_sca_con, Q_abs_con, Q_ext_con, p_con, d_low):
    """ mat_dict: database of all materials in stack
        mat_needle: list of materials that can be inserted as needles (array of strings)
    """
    ml_temp = multilayer(r, n, ban_needle, lam, ml_init.theta, ml_init.phi, ml_init.pol)
    ml_temp.update(r, n)
    init_needle(ml_temp, r, n, lam)
    ml_temp.r = r
    
    n_needle = np.zeros((np.size(lam), np.size(mat_needle))).astype(complex)
    count = 0
    for mat in mat_needle:
        n_needle[:,count] = mat_dict[mat]
        count += 1
    
    loc = dict()
    dMF = dict()
    dMF_min = np.zeros(mat_needle.size)
    for m in range(np.size(mat_needle)):
        nfev = 0
        loc_temp = np.array([0,r[0]])
        l_dMF = deriv_tmm.needle_derivative_wrapper(loc_temp[0], n_needle[:,m], ban_needle, Q_sca, Q_abs, Q_ext, p,
                                                    Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                    lam, ml_temp.theta, ml_temp.phi, ml_temp.lmax, ml_temp.k,
                                                    ml_temp.TE, ml_temp.TM, r, n, ml_temp.pi_l, ml_temp.tau_l,
                                                    ml_temp.Tcu_El, ml_temp.Tcu_Ml, ml_temp.Tcl_El, ml_temp.Tcl_Ml,
                                                    ml_temp.T11_El, ml_temp.T21_El, ml_temp.T11_Ml, ml_temp.T21_Ml,
                                                    ml_temp.t_El, ml_temp.t_Ml, ml_temp.S1, ml_temp.S2, d_low=d_low)
        r_dMF = deriv_tmm.needle_derivative_wrapper(loc_temp[-1], n_needle[:,m], ban_needle, Q_sca, Q_abs, Q_ext, p,
                                                    Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                    lam, ml_temp.theta, ml_temp.phi, ml_temp.lmax, ml_temp.k,
                                                    ml_temp.TE, ml_temp.TM, r, n, ml_temp.pi_l, ml_temp.tau_l,
                                                    ml_temp.Tcu_El, ml_temp.Tcu_Ml, ml_temp.Tcl_El, ml_temp.Tcl_Ml,
                                                    ml_temp.T11_El, ml_temp.T21_El, ml_temp.T11_Ml, ml_temp.T21_Ml,
                                                    ml_temp.t_El, ml_temp.t_Ml, ml_temp.S1, ml_temp.S2, d_low=d_low)
        dMF_temp = np.array([l_dMF,r_dMF]).astype(np.float64)
        result = minimize_scalar(deriv_tmm.needle_derivative_wrapper,
                                 args=(n_needle[:,m], ban_needle, Q_sca, Q_abs, Q_ext, p,
                                       Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                       lam, ml_temp.theta, ml_temp.phi, ml_temp.lmax, ml_temp.k,
                                       ml_temp.TE, ml_temp.TM, r, n, ml_temp.pi_l, ml_temp.tau_l,
                                       ml_temp.Tcu_El, ml_temp.Tcu_Ml, ml_temp.Tcl_El, ml_temp.Tcl_Ml,
                                       ml_temp.T11_El, ml_temp.T21_El, ml_temp.T11_Ml, ml_temp.T21_Ml,
                                       ml_temp.t_El, ml_temp.t_Ml, ml_temp.S1, ml_temp.S2, d_low),
                                 bounds=loc_temp, method='bounded')
        nfev += result.nfev
        if np.abs(result.x - loc_temp[0]) >= 1 and np.abs(result.x - loc_temp[-1]) >= 1:
            loc_temp = np.insert(loc_temp, 1, result.x)
            dMF_temp = np.insert(dMF_temp, 1, result.fun)
        evaluate = np.ones(np.size(loc_temp) - 1)
        while True:
            intervals = np.size(loc_temp)
            loc_copy = loc_temp.copy()
            cnt = 0
            for i in range(intervals-2, -1, -1):
                if evaluate[i]:
                    result = minimize_scalar(deriv_tmm.needle_derivative_wrapper,
                                             args=(n_needle[:,m], ban_needle, Q_sca, Q_abs, Q_ext, p,
                                                   Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                   lam, ml_temp.theta, ml_temp.phi, ml_temp.lmax, ml_temp.k,
                                                   ml_temp.TE, ml_temp.TM, r, n, ml_temp.pi_l, ml_temp.tau_l,
                                                   ml_temp.Tcu_El, ml_temp.Tcu_Ml, ml_temp.Tcl_El, ml_temp.Tcl_Ml,
                                                   ml_temp.T11_El, ml_temp.T21_El, ml_temp.T11_Ml, ml_temp.T21_Ml,
                                                   ml_temp.t_El, ml_temp.t_Ml, ml_temp.S1, ml_temp.S2, d_low),
                                             bounds=loc_copy[i:i+2], method='bounded')
                    nfev += result.nfev
                    if np.abs(result.x - loc_temp[i]) >= 1 and np.abs(result.x - loc_temp[i+1]) >= 1:
                        loc_temp = np.insert(loc_temp, i+1, result.x)
                        dMF_temp = np.insert(dMF_temp, i+1, result.fun)
                        evaluate = np.insert(evaluate, i+1, 1)
                        cnt += 1
                    else:
                        evaluate[i] = 0
            if cnt == 0:
                break
        loc[m] = loc_temp
        dMF[m] = dMF_temp
        dMF_min[m] = np.min(dMF_temp)
        
    if np.min(dMF_min) > -1e-6:
        needle_status = 0
    else:
        needle_status = 1
        close_to_boundary = np.zeros(0).astype(bool)
        for m in range(mat_needle.size):
            for z in range(loc[m].size):
                z_final = loc[m][z]
                close_to_boundary = np.append(close_to_boundary, np.sum(np.abs(r - z_final) < 1) > 0)
        if np.sum(close_to_boundary) == close_to_boundary.size:
            needle_status = 0
        
    return needle_status, n_needle, loc, dMF

def deep_search(iteration, ml_init, mat_needle, n_needle, loc, dMF, r, n, ban_needle,
                Q_sca_con, Q_abs_con, Q_ext_con, p_con, d_low):
    MF_deep = np.zeros(mat_needle.size)
    indMF_deep = np.zeros(mat_needle.size)
    r_out = dict()
    n_out = dict()
    ban_needle_out = dict()
    for m in range(mat_needle.size):
        MF_deep_temp = np.zeros(loc[m].size)
        for z in range(loc[m].size):
            if dMF[m][z] > 0: # skip if needle gradient is positive
                MF_deep_temp[z] = np.nan
                continue
            
            z_final = loc[m][z]
            
            close_to_boundary = 0
            for l in range(r.size):
                if np.abs(r[l] - z_final) < 1 or z_final < 1:
                    close_to_boundary = 1
                
            if close_to_boundary == 1: # skip if needle is too close to an existing boundary
                MF_deep_temp[z] = np.nan
                continue
                
            if z_final < r[-1]:
                r_final = r.size
            elif z_final > r[1]:
                r_final = 1
            else:
                for l in range(1, r.size-1):
                    if z_final < r[l] and z_final > r[l+1]:
                        r_final = l + 1
                        break
        
            n_new = np.concatenate((n[:,:r_final+1], n_needle[:,m].reshape((np.size(ml_init.lam),1)), n[:,r_final:]), axis=1)
            r_new = np.hstack((r[:r_final], z_final + 1e-3, z_final - 1e-3, r[r_final:]))
            ban_needle_new = np.hstack((ban_needle[:r_final], False, ban_needle[r_final-1:]))
            
            r_new, MF_deep_temp[z], Q_sca_new, Q_abs_new, Q_ext_new, p_new = refine_r(iteration, ml_init, r_new, n_new, ml_init.lam,
                                                                                      Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                                      d_low=d_low)
            r_out[m,z] = r_new
            n_out[m,z] = n_new
            ban_needle_out[m,z] = ban_needle_new
        if np.sum(np.isnan(MF_deep_temp)) == MF_deep_temp.size:
            MF_deep[m] = np.nan
            indMF_deep[m] = 0
        else:
            MF_deep[m] = np.nanmin(MF_deep_temp)
            indMF_deep[m] = np.nanargmin(MF_deep_temp)

    if np.sum(np.isnan(MF_deep)) == MF_deep.size:
        ml_init.update(r, n)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                              ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi,
                                                              ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                              ml_init.eta_tilde)
        nancheck = np.isnan(np.sum(t_El, axis=0))
        if np.sum(nancheck) > 0:
            lmax = np.min(np.argwhere(nancheck)) - 1
            ml_init.update(r, n, lmax=lmax)
            Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                                  ml_init.TE, ml_init.TM, r, n, ml_init.psi, ml_init.dpsi,
                                                                  ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                                  ml_init.eta_tilde)
        
        plot_spectra(iteration, ml_init.lam, ml_init.theta, ml_init.phi, Q_sca, Q_abs, Q_ext, p,
                     Q_sca_con, Q_abs_con, Q_ext_con, p_con, r, n)
        return n, r, ban_needle, Q_sca, Q_abs, Q_ext, p
    else:
        mat_final = np.nanargmin(MF_deep)
        n_out = n_out[mat_final,indMF_deep[mat_final]]
        r_out = r_out[mat_final,indMF_deep[mat_final]]
        ml_init.update(r_out, n_out)
        Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                              ml_init.TE, ml_init.TM, r_out, n_out, ml_init.psi, ml_init.dpsi,
                                                              ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                              ml_init.eta_tilde)
        nancheck = np.isnan(np.sum(t_El, axis=0))
        if np.sum(nancheck) > 0:
            lmax = np.min(np.argwhere(nancheck)) - 1
            ml_init.update(r_out, n_out, lmax=lmax)
            Q_sca, Q_abs, Q_ext, p, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam, ml_init.theta, ml_init.phi, ml_init.lmax, ml_init.k,
                                                                  ml_init.TE, ml_init.TM, r_out, n_out, ml_init.psi, ml_init.dpsi,
                                                                  ml_init.ksi, ml_init.dksi, ml_init.pi_l, ml_init.tau_l,
                                                                  ml_init.eta_tilde)
        
        plot_spectra(iteration, ml_init.lam, ml_init.theta, ml_init.phi, Q_sca, Q_abs, Q_ext, p,
                     Q_sca_con, Q_abs_con, Q_ext_con, p_con, r_out, n_out)
        return n_out, r_out, ban_needle_out[mat_final,indMF_deep[mat_final]], Q_sca, Q_abs, Q_ext, p

def run_needle(index, mat_dict, mat_needle, r, n, ban_needle, lam, theta, phi, pol, Q_sca_con, Q_abs_con, Q_ext_con, p_con,
               d_low=10):
    ml_init = multilayer(r, n, ban_needle, lam, theta, phi, pol)
    ml_init.update(r, n)

    iteration = 1
    r_new, cost, Q_sca_new, Q_abs_new, Q_ext_new, p_new = refine_r(index, ml_init, r, n, lam, Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                   d_low=d_low)
    n_new = n.copy()
    ban_needle_new = ban_needle.copy()
    needle_status = 1
    cost_count = 0
    cost_prev = np.inf
    while True:
        iteration += 1
        needle_status, n_needle, loc, dMF = insert_needle(ml_init, mat_dict, mat_needle, r_new, n_new, ban_needle_new, lam,
                                                          Q_sca_new, Q_abs_new, Q_ext_new, p_new,
                                                          Q_sca_con, Q_abs_con, Q_ext_con, p_con, d_low)
        if needle_status == 0:
            break
        
        n_new, r_new, ban_needle_new, Q_sca_new, Q_abs_new, Q_ext_new, p_new = deep_search(index, ml_init, mat_needle, n_needle,
                                                                                           loc, dMF, r_new, n_new, ban_needle_new,
                                                                                           Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                                           d_low=d_low)
        
        if cost < 0.99*cost_prev:
            cost_prev = cost
        else:
            cost_count += 1
        if cost_count > 1:
            break
    iteration += 1
    
    # Clean up layers that are too thin
    thin_layer = 1
    while thin_layer:
        r_fin = r_new[0]
        n_fin = n_new[:,0].reshape(np.size(lam), 1)
        thin_layer = 0
        for l in range(r_new.size-1):
            if r_new[l]-r_new[l+1] > d_low:
                r_fin = np.append(r_fin, r_new[l+1])
                n_fin = np.concatenate((n_fin, n_new[:,l+1].reshape(np.size(lam), 1)), axis=1)
            else:
                thin_layer = 1
        if r_new[-1] > d_low:
            n_fin = np.concatenate((n_fin, n_new[:,-1].reshape(np.size(lam), 1)), axis=1)
        elif r_fin.size != 1:
            r_fin = r_fin[:-1]
            thin_layer = 1
        else:
            Q_sca_fin = Q_sca_new.copy()
            Q_abs_fin = Q_abs_new.copy()
            Q_ext_fin = Q_ext_new.copy()
            p_fin = p_new.copy()
            break
        
        if r_fin.size > 1:
            for l in range(r_fin.size - 1, -1, -1):
                if np.array_equal(n_fin[:,l+1], n_fin[:,l]):
                    n_fin = np.delete(n_fin, l+1, axis=1)
                    r_fin = np.delete(r_fin, l)
        r_new, cost, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin = refine_r(index, ml_init, r_fin, n_fin, lam,
                                                                       Q_sca_con, Q_abs_con, Q_ext_con, p_con,
                                                                       d_low=d_low)
        n_new = n_fin.copy()
    
    r_fin = r_new.copy()
    n_fin = n_new.copy()
    
    return r_fin, n_fin, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, cost

def plot_spectra(iteration, lam, theta, phi, Q_sca, Q_abs, Q_ext, p, Q_sca_con, Q_abs_con, Q_ext_con, p_con, r, n):
    fig, ax = plt.subplots(figsize=[12,5], dpi=100)
    ax.plot(lam, Q_sca, linewidth=1, color='darkblue', label='C_sca')
    ax.plot(lam, Q_abs, linewidth=1, color='darkred', label='C_abs')
    ax.plot(lam, Q_ext, linewidth=1, color='goldenrod', label='C_ext')
    ax.plot(lam, Q_sca_con[0,0,:], linewidth=1, linestyle='dashed', color='darkblue',
            marker='.', markeredgecolor='darkblue', markerfacecolor='none', label='C_sca')
    ax.plot(lam, Q_abs_con[0,0,:], linewidth=1, linestyle='dashed', color='darkred',
            marker='.', markeredgecolor='darkred', markerfacecolor='none', label='C_abs')
    ax.plot(lam, Q_ext_con[0,0,:], linewidth=1, linestyle='dashed', color='goldenrod',
            marker='.', markeredgecolor='goldenrod', markerfacecolor='none', label='C_ext')
    Q_max = np.nanmax((np.max(Q_sca), np.max(Q_abs), np.max(Q_ext),
                       np.nanmax(Q_sca_con[0,0,:]), np.nanmax(Q_abs_con[0,0,:]), np.nanmax(Q_ext_con[0,0,:])))
    ax.set_ylim(-0.1*Q_max, 1.1*Q_max)
    ax.set_xlabel('Wavelength (nm)')
    ax.legend()
    plt.savefig(directory + '\\efficiencies_' + str(iteration))
    plt.close()
    
    fig, ax = plt.subplots(2, 2, figsize=[12,10], dpi=100)
    xgrid, ygrid = np.meshgrid(lam, theta*180/np.pi, indexing='ij')
    p_plot = p.copy()
    p_con_plot = p_con.copy()
    for w in range(lam.size):
        for n_p in range(phi.size):
            p_plot[w,:,n_p] /= np.max(p_plot[w,:,n_p])
            p_con_plot[0,0,w,:,n_p] /= np.nanmax(p_con_plot[0,0,w,:,n_p])
    vmax = 1 #np.max((np.max(p), np.nanmax(p_con[0,:,:,:])))
    vmin = 0 #np.min((np.max(p), np.nanmax(p_con[0,:,:,:])))
    im1 = ax[0,0].contourf(xgrid, ygrid, p_plot[:,:,0], cmap='plasma', vmax=vmax, vmin=vmin, levels=100)
    im2 = ax[0,1].contourf(xgrid, ygrid, p_con_plot[0,0,:,:,0], cmap='plasma', vmax=vmax, vmin=vmin, levels=100)
    im3 = ax[1,0].contourf(xgrid, ygrid, p_plot[:,:,1], cmap='plasma', vmax=vmax, vmin=vmin, levels=100)
    im4 = ax[1,1].contourf(xgrid, ygrid, p_con_plot[0,0,:,:,1], cmap='plasma', vmax=vmax, vmin=vmin, levels=100)
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.85, wspace=0.15, hspace=0.2)
    c_ax = fig.add_axes([0.91, 0.05, 0.015, 0.9])
    fig.colorbar(im1, cax=c_ax)
    ax[0,0].set_xlim(lam[0], lam[-1])
    ax[0,1].set_xlim(lam[0], lam[-1])
    ax[1,0].set_xlim(lam[0], lam[-1])
    ax[1,1].set_xlim(lam[0], lam[-1])
    ax[0,0].set_ylim(theta[0]*180/np.pi, theta[-1]*180/np.pi)
    ax[0,1].set_ylim(theta[0]*180/np.pi, theta[-1]*180/np.pi)
    ax[1,0].set_ylim(theta[0]*180/np.pi, theta[-1]*180/np.pi)
    ax[1,1].set_ylim(theta[0]*180/np.pi, theta[-1]*180/np.pi)
    ax[1,0].set_xlabel('Wavelength (nm)')
    ax[1,1].set_xlabel('Wavelength (nm)')
    ax[0,0].set_ylabel('Azimuthal Angle (deg.)')
    ax[1,0].set_ylabel('Azimuthal Angle (deg.)')
    ax[0,0].set_title('TE Phase Fct.')
    ax[0,1].set_title('TE Target')
    ax[1,0].set_title('TM Phase Fct.')
    ax[1,1].set_title('TM Target')
    plt.savefig(directory + '\\phase_fct_' + str(iteration))
    plt.close()
    
    fig, ax = plt.subplots(figsize=[12,5], dpi=100)
    midpt = int(np.size(lam)/2)
    ax.hlines(np.real(n[midpt,-1]), 0, r[-1], linewidth=1, color='black')
    ax.vlines(np.sum(r[-1]), np.min((np.real(n[midpt,-1]), np.real(n[midpt,-2]))), np.max((np.real(n[midpt,-1]), np.real(n[midpt,-2]))),
              linewidth=1, color='black')
    for l in range(np.size(r)-2, -1, -1):
        ax.hlines(np.real(n[midpt,l+1]), r[l+1], r[l], linewidth=1, color='black')
        ax.vlines(r[l], np.min((np.real(n[midpt,l+1]), np.real(n[midpt,l]))), np.max((np.real(n[midpt,l+1]), np.real(n[midpt,l]))),
                  linewidth=1, color='black')
    ax.hlines(np.real(n[midpt,0]), r[0], r[0]*1.1, linewidth=1, linestyle='dashed', color='black')
    ax.set_xlim(0, r[0]*1.05)
    ax.set_xlabel('Radial Distance (nm)')
    plt.savefig(directory + '\\n_profile_' + str(iteration))
    plt.close()