import numpy as np
import eschallot.mie.special_functions as spec
import eschallot.mie.tmm_mie as tmm
import eschallot.mie.simulate_particle as sim
import eschallot.optimization.cost_gradients as cg
import eschallot.util.read_mat_data as rmd
from scipy.optimize import minimize_scalar, minimize, LinearConstraint, Bounds
from mpi4py import MPI
comm = MPI.COMM_WORLD

class multilayer:
    def __init__(self, lam, theta, phi):
        """ n: refractive index, dim --> layer(outer to inner) x lam(short to long)
        r: interface radius, dim --> layer(outer to inner) x 1 (excluding external medium)
        lam: wavelength, dim --> 1 x lam(short to long)
        theta: incident angle, dim --> 1 x 1 """
        
        # User-provided multilayer quantities
        self.lam = lam
        self.theta = theta
        self.phi = phi

        # Derived quantities
        self.k = (2*np.pi)/lam

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
        
        self.ksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128) # First index: 0 --> no tilde, 1 --> tilde
        self.dksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.d2ksi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.psi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.dpsi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)
        self.d2psi = np.zeros((2, wvl, self.lmax+1, layer)).astype(np.complex128)

        self.x = n[:,1:]*self.k[:,np.newaxis]*r[np.newaxis,:]
        self.eta_tilde = n[:,1:]/n[:,:-1]
        self.x_tilde = self.x/self.eta_tilde
    
        self.ksi[0,:,:,:], self.dksi[0,:,:,:] = spec.RB_ksi(self.x, self.lmax)
        self.psi[0,:,:,:], self.dpsi[0,:,:,:] = spec.RB_psi(self.x, self.lmax)
        self.ksi[1,:,:,:], self.dksi[1,:,:,:] = spec.RB_ksi(self.x_tilde, self.lmax)
        self.psi[1,:,:,:], self.dpsi[1,:,:,:] = spec.RB_psi(self.x_tilde, self.lmax)
        
        coeff = np.zeros((wvl, self.lmax+1, layer))
        for n_l in range(self.lmax+1):
            coeff[:,n_l,:] = n_l*(n_l + 1)

        self.d2ksi[0,:,:,:] = -(1 - coeff/self.x[:,np.newaxis,:]**2)*self.ksi[0,:,:,:]
        self.d2psi[0,:,:,:] = -(1 - coeff/self.x[:,np.newaxis,:]**2)*self.psi[0,:,:,:]
        self.d2ksi[1,:,:,:] = -(1 - coeff/self.x_tilde[:,np.newaxis,:]**2)*self.ksi[1,:,:,:]
        self.d2psi[1,:,:,:] = -(1 - coeff/self.x_tilde[:,np.newaxis,:]**2)*self.psi[1,:,:,:]

def refine_r(index,
             ml,
             r0,
             n,
             lam,
             d_low,
             r_max,
             custom_cost,
             ):

    ub = np.inf*np.ones(r0.size)
    ub[0] = r_max
    bnd = Bounds(np.zeros(r0.size), ub)

    A = np.zeros((r0.size, r0.size))
    for l in range(r0.size):
        A[l,l] = 1
        if l < r0.size - 1:
            A[l,l+1] = -1
    constr = LinearConstraint(A, lb=np.ones(r0.size), ub=np.inf*np.ones(r0.size))
    
    verbose = 0
    result = minimize(cg.cost, r0, args=(n, index, lam, ml, custom_cost),
                      method='trust-constr', jac=cg.shape_gradient, constraints=constr, bounds=bnd,
                      options={'verbose': verbose, 'gtol': 1e-8, 'xtol': 1e-8, 'maxiter': 1000})

    r_new = result.x.copy()
    cost = result.fun        

    Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
        S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(lam,
                                                          ml.theta,
                                                          ml.phi,
                                                          ml.lmax,
                                                          ml.k,
                                                          result.x,
                                                          n,
                                                          ml.psi,
                                                          ml.dpsi,
                                                          ml.ksi,
                                                          ml.dksi,
                                                          ml.pi_l,
                                                          ml.tau_l,
                                                          ml.eta_tilde)                                                              
    
    return r_new, cost, Q_sca, Q_abs, Q_ext, p, diff_CS

def init_needle(ml, r, n, lam):
    # make original object for reference in grad_needle
    result = tmm.transfer_matrix(lam,
                                 ml.theta,
                                 ml.phi,
                                 ml.lmax,
                                 ml.k,
                                 r,
                                 n,
                                 ml.psi,
                                 ml.dpsi,
                                 ml.ksi,
                                 ml.dksi,
                                 ml.pi_l,
                                 ml.tau_l,
                                 ml.eta_tilde)

    nancheck = np.isnan(np.sum(result[8], axis=0))
    lmax = None
    if np.sum(nancheck) > 0:
        lmax = np.min(np.argwhere(nancheck)) - 1
        ml.update(r, n, lmax=lmax)
        result = tmm.transfer_matrix(lam,
                                     ml.theta,
                                     ml.phi,
                                     ml.lmax,
                                     ml.k,
                                     r,
                                     n,
                                     ml.psi,
                                     ml.dpsi,
                                     ml.ksi,
                                     ml.dksi,
                                     ml.pi_l,
                                     ml.tau_l,
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

def insert_needle(ml_init,
                  mat_dict,
                  mat_needle,
                  r,
                  n,
                  ban_needle,
                  lam,
                  Q_sca,
                  Q_abs,
                  Q_ext,
                  p,
                  diff_CS,
                  d_low,
                  custom_cost,
                  ):
    """ mat_dict: database of all materials in the multilayer stack
        mat_needle: list of materials that can be inserted as needles (array of strings)
    """
    ml_temp = multilayer(lam, ml_init.theta, ml_init.phi)
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
        l_dMF = cg.topology_gradient(loc_temp[0],
                                     n_needle[:,m],
                                     ban_needle,
                                     Q_sca,
                                     Q_abs,
                                     Q_ext,
                                     p,
                                     diff_CS,
                                     lam,
                                     ml_temp.theta,
                                     ml_temp.phi,
                                     ml_temp.lmax,
                                     ml_temp.k,
                                     r,
                                     n,
                                     ml_temp.pi_l,
                                     ml_temp.tau_l,
                                     ml_temp.Tcu_El,
                                     ml_temp.Tcu_Ml,
                                     ml_temp.Tcl_El,
                                     ml_temp.Tcl_Ml,
                                     ml_temp.T11_El,
                                     ml_temp.T21_El,
                                     ml_temp.T11_Ml,
                                     ml_temp.T21_Ml,
                                     ml_temp.t_El,
                                     ml_temp.t_Ml,
                                     ml_temp.S1,
                                     ml_temp.S2,
                                     d_low,
                                     custom_cost)
                                                    
        r_dMF = cg.topology_gradient(loc_temp[-1],
                                     n_needle[:,m],
                                     ban_needle,
                                     Q_sca,
                                     Q_abs,
                                     Q_ext,
                                     p,
                                     diff_CS,
                                     lam,
                                     ml_temp.theta,
                                     ml_temp.phi,
                                     ml_temp.lmax,
                                     ml_temp.k,
                                     r,
                                     n,
                                     ml_temp.pi_l,
                                     ml_temp.tau_l,
                                     ml_temp.Tcu_El,
                                     ml_temp.Tcu_Ml,
                                     ml_temp.Tcl_El,
                                     ml_temp.Tcl_Ml,
                                     ml_temp.T11_El,
                                     ml_temp.T21_El,
                                     ml_temp.T11_Ml,
                                     ml_temp.T21_Ml,
                                     ml_temp.t_El,
                                     ml_temp.t_Ml,
                                     ml_temp.S1,
                                     ml_temp.S2,
                                     d_low,
                                     custom_cost)
                                                    
        dMF_temp = np.array([l_dMF,r_dMF]).astype(np.float64)
        result = minimize_scalar(cg.topology_gradient,
                                 args=(n_needle[:,m],
                                       ban_needle,
                                       Q_sca,
                                       Q_abs,
                                       Q_ext,
                                       p,
                                       diff_CS,
                                       lam,
                                       ml_temp.theta,
                                       ml_temp.phi,
                                       ml_temp.lmax,
                                       ml_temp.k,
                                       r,
                                       n,
                                       ml_temp.pi_l,
                                       ml_temp.tau_l,
                                       ml_temp.Tcu_El,
                                       ml_temp.Tcu_Ml,
                                       ml_temp.Tcl_El,
                                       ml_temp.Tcl_Ml,
                                       ml_temp.T11_El,
                                       ml_temp.T21_El,
                                       ml_temp.T11_Ml,
                                       ml_temp.T21_Ml,
                                       ml_temp.t_El,
                                       ml_temp.t_Ml,
                                       ml_temp.S1,
                                       ml_temp.S2,
                                       d_low,
                                       custom_cost),
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
                    result = minimize_scalar(cg.topology_gradient,
                                             args=(n_needle[:,m],
                                                   ban_needle,
                                                   Q_sca,
                                                   Q_abs,
                                                   Q_ext,
                                                   p,
                                                   diff_CS,
                                                   lam,
                                                   ml_temp.theta,
                                                   ml_temp.phi,
                                                   ml_temp.lmax,
                                                   ml_temp.k,
                                                   r,
                                                   n,
                                                   ml_temp.pi_l,
                                                   ml_temp.tau_l,
                                                   ml_temp.Tcu_El,
                                                   ml_temp.Tcu_Ml,
                                                   ml_temp.Tcl_El,
                                                   ml_temp.Tcl_Ml,
                                                   ml_temp.T11_El,
                                                   ml_temp.T21_El,
                                                   ml_temp.T11_Ml,
                                                   ml_temp.T21_Ml,
                                                   ml_temp.t_El,
                                                   ml_temp.t_Ml,
                                                   ml_temp.S1,
                                                   ml_temp.S2,
                                                   d_low,
                                                   custom_cost),
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
        loc[m] = loc_temp[np.argsort(dMF_temp)]
        dMF[m] = dMF_temp[np.argsort(dMF_temp)]
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

def deep_search(index,
                ml_init,
                mat_needle,
                n_needle,
                loc,
                dMF,
                mat_profile,
                r,
                n,
                ban_needle,
                d_low,
                r_max,
                custom_cost,
                ):
                
    MF_deep = np.zeros(mat_needle.size)
    indMF_deep = np.zeros(mat_needle.size)
    r_out = dict()
    n_out = dict()
    ban_needle_out = dict()
    mat_profile_out = dict()
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
            r_new = np.hstack((r[:r_final], z_final + 1e-1, z_final - 1e-1, r[r_final:]))
            ban_needle_new = np.hstack((ban_needle[:r_final], False, ban_needle[r_final-1:]))
            mat_profile_new = np.hstack((mat_profile[:r_final+1], mat_needle[m], mat_profile[r_final:]))
            
            r_new, MF_deep_temp[z], Q_sca_new, Q_abs_new, Q_ext_new, p_new, diff_CS_new = refine_r(index,
                                                                                                   ml_init,
                                                                                                   r_new,
                                                                                                   n_new,
                                                                                                   ml_init.lam,
                                                                                                   d_low,
                                                                                                   r_max,
                                                                                                   custom_cost)
            r_out[m,z] = r_new
            n_out[m,z] = n_new
            ban_needle_out[m,z] = ban_needle_new
            mat_profile_out[m,z] = mat_profile_new
        if np.sum(np.isnan(MF_deep_temp)) == MF_deep_temp.size:
            MF_deep[m] = np.nan
            indMF_deep[m] = 0
        else:
            MF_deep[m] = np.nanmin(MF_deep_temp)
            indMF_deep[m] = np.nanargmin(MF_deep_temp)

    if np.sum(np.isnan(MF_deep)) == MF_deep.size:
        ml_init.update(r, n)
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam,
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
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam,
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

        return n, r, ban_needle, mat_profile, Q_sca, Q_abs, Q_ext, p, diff_CS
    else:
        mat_final = np.nanargmin(MF_deep)
        n_out = n_out[mat_final,indMF_deep[mat_final]]
        r_out = r_out[mat_final,indMF_deep[mat_final]]
        ml_init.update(r_out, n_out)
        Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
            S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam,
                                                              ml_init.theta,
                                                              ml_init.phi,
                                                              ml_init.lmax,
                                                              ml_init.k,
                                                              r_out,
                                                              n_out,
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
            ml_init.update(r_out, n_out, lmax=lmax)
            Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = tmm.efficiencies(ml_init.lam,
                                                                  ml_init.theta,
                                                                  ml_init.phi,
                                                                  ml_init.lmax,
                                                                  ml_init.k,
                                                                  r_out,
                                                                  n_out,
                                                                  ml_init.psi,
                                                                  ml_init.dpsi,
                                                                  ml_init.ksi,
                                                                  ml_init.dksi,
                                                                  ml_init.pi_l,
                                                                  ml_init.tau_l,
                                                                  ml_init.eta_tilde)
        
        return n_out, r_out, ban_needle_out[mat_final,indMF_deep[mat_final]], mat_profile_out[mat_final,indMF_deep[mat_final]], Q_sca, Q_abs, Q_ext, p, diff_CS

def run_needle(index,
               mat_dict_cost,
               mat_dict_plot,
               mat_needle,
               mat_profile,
               r,
               n,
               ban_needle,
               lam_cost,
               theta_cost,
               phi_cost,
               lam_plot,
               theta_plot,
               phi_plot,
               d_low,
               r_max,
               max_layers,
               custom_cost,
               ):
               
    ml_init = multilayer(lam_cost, theta_cost, phi_cost)
    ml_init.update(r, n)

    iteration = 1
    r_new, cost, Q_sca_new, Q_abs_new, Q_ext_new, p_new, diff_CS_new = refine_r(index,
                                                                                ml_init,
                                                                                r,
                                                                                n,
                                                                                lam_cost,
                                                                                d_low,
                                                                                r_max,
                                                                                custom_cost)
    
    mat_profile_new = mat_profile.copy()
    n_new = n.copy()
    ban_needle_new = ban_needle.copy()
    needle_status = 1
    while True:
        iteration += 1
        needle_status, n_needle, loc, dMF = insert_needle(ml_init,
                                                          mat_dict_cost,
                                                          mat_needle,
                                                          r_new,
                                                          n_new,
                                                          ban_needle_new,
                                                          lam_cost,
                                                          Q_sca_new,
                                                          Q_abs_new,
                                                          Q_ext_new,
                                                          p_new,
                                                          diff_CS_new,
                                                          d_low,
                                                          custom_cost)
        if needle_status == 0:
            break
        
        n_new, r_new, ban_needle_new, mat_profile_new,\
            Q_sca_new, Q_abs_new, Q_ext_new, p_new, diff_CS_new = deep_search(index,
                                                                              ml_init,
                                                                              mat_needle,
                                                                              n_needle,
                                                                              loc,
                                                                              dMF,
                                                                              mat_profile_new,
                                                                              r_new,
                                                                              n_new,
                                                                              ban_needle_new,
                                                                              d_low,
                                                                              r_max,
                                                                              custom_cost)
        
        thickness = r_new[:-1] - r_new[1:]
        if np.sum(thickness < d_low) > 1:
            break
        if max_layers is not None:
            if r_new.size >= max_layers:
                break
                
    iteration += 1
    
    # Clean up layers that are too thin
    thin_layer = 1
    while thin_layer:
        r_fin = r_new[0]
        n_fin = n_new[:,0].reshape(np.size(lam_cost), 1)
        mat_profile_fin = mat_profile_new[0]
        thin_layer = 0
        for l in range(r_new.size-1):
            if r_new[l]-r_new[l+1] > d_low:
                r_fin = np.append(r_fin, r_new[l+1])
                n_fin = np.concatenate((n_fin, n_new[:,l+1].reshape(np.size(lam_cost), 1)), axis=1)
                mat_profile_fin = np.append(mat_profile_fin, mat_profile_new[l+1])
            else:
                thin_layer = 1
        if r_new[-1] > d_low:
            n_fin = np.concatenate((n_fin, n_new[:,-1].reshape(np.size(lam_cost), 1)), axis=1)
            mat_profile_fin = np.append(mat_profile_fin, mat_profile_new[-1])
        elif r_fin.size != 1:
            r_fin = r_fin[:-1]
            thin_layer = 1
        else:
            Q_sca_fin = Q_sca_new.copy()
            Q_abs_fin = Q_abs_new.copy()
            Q_ext_fin = Q_ext_new.copy()
            p_fin = p_new.copy()
            diff_CS_fin = diff_CS_new.copy()
            break
        
        if r_fin.size > 1:
            for l in range(r_fin.size - 1, -1, -1):
                if np.array_equal(n_fin[:,l+1], n_fin[:,l]):
                    n_fin = np.delete(n_fin, l+1, axis=1)
                    r_fin = np.delete(r_fin, l)
                    mat_profile_fin = np.delete(mat_profile_fin, l+1)
        r_new, cost, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, diff_CS_fin = refine_r(index,
                                                                                    ml_init,
                                                                                    r_fin,
                                                                                    n_fin,
                                                                                    lam_cost,
                                                                                    d_low,
                                                                                    r_max,
                                                                                    custom_cost)
        n_new = n_fin.copy()
        mat_profile_new = mat_profile_fin.copy()
    
    r_fin = r_new.copy()
    n_fin = n_new.copy()
    mat_profile_fin = mat_profile_new.copy()
    
    # High-Resolution Phase Function Computation
    n_fin = np.zeros((np.size(lam_plot,0), np.size(mat_profile_fin,0))).astype(complex)
    count = 0
    for mat in mat_profile_fin:
        n_fin[:,count] = mat_dict_plot[mat]
        count += 1
    
    Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, diff_CS_fin, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
                S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(lam_plot, theta_plot, phi_plot, r_fin, n_fin)
    
    return r_fin, n_fin, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, diff_CS_fin, cost

def radius_sweep(output_filename,
                 r_min,
                 r_max,
                 N_sweep,
                 d_low,
                 max_layers,
                 mat_profile,
                 mat_needle,
                 lam_cost,
                 theta_cost,
                 phi_cost,
                 lam_plot,
                 theta_plot,
                 phi_plot,
                 custom_cost,
                 mat_data_dir=None,
                 ):

    # Create n
    mat_type = list(set(np.hstack((mat_profile, mat_needle))))
    raw_wavelength, mat_dict_cost_default = rmd.load_all(lam_cost, 'n_k', mat_type)
    raw_wavelength, mat_dict_plot_default = rmd.load_all(lam_plot, 'n_k', mat_type)
    
    if mat_data_dir is not None:
        raw_wavelength, mat_dict_cost_custom = rmd.load_all(lam_cost, 'n_k', mat_type, directory=mat_data_dir)
        raw_wavelength, mat_dict_plot_custom = rmd.load_all(lam_plot, 'n_k', mat_type, directory=mat_data_dir)
    else:
        mat_dict_cost_custom = dict()
        mat_dict_plot_custom = dict()
    
    mat_dict_cost = {**mat_dict_cost_default, **mat_dict_cost_custom}
    mat_dict_plot = {**mat_dict_plot_default, **mat_dict_plot_custom}
    
    n = np.zeros((np.size(lam_cost,0), np.size(mat_profile,0))).astype(complex)
    count = 0
    for mat in mat_profile:
        n[:,count] = mat_dict_cost[mat]
        count += 1

    ### Initial Single-Layer Optimization to Eliminate Redundant Runs
    # Distribute Radii for Sweeping
    radius_list = np.linspace(r_min, r_max, N_sweep) # in nm
    
    quo, rem = divmod(N_sweep, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size+1)])
    
    radius_list_proc = radius_list[data_disp[comm.rank]:data_disp[comm.rank+1]]

    # Run Optimization
    if comm.rank == 0:
        print('### Initial Single-Layer Optimization (N_sweep = ' + str(N_sweep) + ')', flush=True)
        print('    Progress: ', end='', flush=True)
    
    ban_needle = np.array([True]) # Outer(excluding embedding medium) to inner
    radius_init_proc = np.zeros(data_size[comm.rank])
    for nr in range(data_size[comm.rank]):
        r_init, _, _, _, _, _, _, _ = run_needle(comm.rank,
                                                 mat_dict_cost,
                                                 mat_dict_plot,
                                                 mat_needle,
                                                 mat_profile,
                                                 np.array([radius_list_proc[nr]]),
                                                 n,
                                                 ban_needle,
                                                 lam_cost,
                                                 theta_cost,
                                                 phi_cost,
                                                 lam_plot,
                                                 theta_plot,
                                                 phi_plot,
                                                 d_low,
                                                 r_max,
                                                 max_layers,
                                                 custom_cost)
        
        assert r_init.size == 1
        
        radius_init_proc[nr] = r_init[0]

    print('/', end='', flush=True)

    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])

    radius_init = np.zeros(N_sweep)
    comm.Allgatherv(radius_init_proc, [radius_init, data_size, data_disp, MPI.DOUBLE])
    
    radius_list = radius_init[0].reshape(1)
    for ns in range(N_sweep-1):
        if ns == 0:
            if np.abs(radius_init[ns+1] - radius_list) > 1e-3*radius_list:
                radius_list = np.append(radius_list, radius_init[ns+1])
        else:
            redundant = np.any(np.abs(radius_init[ns+1] - radius_list) <= 1e-3*radius_init[ns+1])
            if not redundant:
                radius_list = np.append(radius_list, radius_init[ns+1])
    N_sweep = radius_list.size
    
    ### Optimization of Non-Redundant Radii
    # Distribute Radii for Sweeping
    quo, rem = divmod(N_sweep, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size+1)])
    
    np.random.shuffle(radius_list)
    radius_list_proc = radius_list[data_disp[comm.rank]:data_disp[comm.rank+1]]

    # Run Optimization
    if comm.rank == 0:
        print('\n### Topology Optimization (N_candidates = ' + str(N_sweep) + ')', flush=True)
        print('    Progress: ', end='', flush=True)
    
    ban_needle = np.array([False]) # Outer(excluding embedding medium) to inner
    radius_proc = dict()
    RI_proc = dict()
    Q_sca_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    Q_abs_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    Q_ext_proc = np.zeros((data_size[comm.rank], lam_plot.size))
    p_proc = np.zeros((data_size[comm.rank], lam_plot.size, theta_plot.size, phi_plot.size))
    diff_CS_proc = np.zeros((data_size[comm.rank], lam_plot.size, theta_plot.size, phi_plot.size))
    N_layer_proc = np.zeros(data_size[comm.rank])
    cost_proc = np.zeros(data_size[comm.rank])
    for nr in range(data_size[comm.rank]):
        r_fin, n_fin, Q_sca_fin, Q_abs_fin, Q_ext_fin, p_fin, diff_CS_fin, cost_fin = run_needle(comm.rank,
                                                                                                 mat_dict_cost,
                                                                                                 mat_dict_plot,
                                                                                                 mat_needle,
                                                                                                 mat_profile,
                                                                                                 np.array([radius_list_proc[nr]]),
                                                                                                 n,
                                                                                                 ban_needle,
                                                                                                 lam_cost,
                                                                                                 theta_cost,
                                                                                                 phi_cost,
                                                                                                 lam_plot,
                                                                                                 theta_plot,
                                                                                                 phi_plot,
                                                                                                 d_low,
                                                                                                 r_max,
                                                                                                 max_layers,
                                                                                                 custom_cost)
        
        radius_proc[nr] = r_fin
        N_layer_proc[nr] = r_fin.size
        RI_proc[nr] = n_fin
        Q_sca_proc[nr,:] = Q_sca_fin
        Q_abs_proc[nr,:] = Q_abs_fin
        Q_ext_proc[nr,:] = Q_ext_fin
        p_proc[nr,:,:,:] = p_fin
        diff_CS_proc[nr,:,:,:] = diff_CS_fin
        cost_proc[nr] = cost_fin

    print('/', end='', flush=True)

    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])

    N_layer = np.zeros(N_sweep)
    cost = np.zeros(N_sweep)
    comm.Allgatherv(N_layer_proc, [N_layer, data_size, data_disp, MPI.DOUBLE])
    comm.Gatherv(cost_proc, [cost, data_size, data_disp, MPI.DOUBLE], root=0)
        
    data_size_temp = data_size*lam_plot.size
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    Q_sca_temp = np.zeros(N_sweep*lam_plot.size)
    Q_abs_temp = np.zeros(N_sweep*lam_plot.size)
    Q_ext_temp = np.zeros(N_sweep*lam_plot.size)
    comm.Gatherv(Q_sca_proc.reshape(-1), [Q_sca_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(Q_abs_proc.reshape(-1), [Q_abs_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(Q_ext_proc.reshape(-1), [Q_ext_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    Q_sca = Q_sca_temp.reshape(N_sweep, lam_plot.size)
    Q_abs = Q_abs_temp.reshape(N_sweep, lam_plot.size)
    Q_ext = Q_ext_temp.reshape(N_sweep, lam_plot.size)
    
    data_size_temp = data_size*lam_plot.size*theta_plot.size*phi_plot.size
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    p_temp = np.zeros(N_sweep*lam_plot.size*theta_plot.size*phi_plot.size)
    diff_CS_temp = np.zeros(N_sweep*lam_plot.size*theta_plot.size*phi_plot.size)
    comm.Gatherv(p_proc.reshape(-1), [p_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(diff_CS_proc.reshape(-1), [diff_CS_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    p = p_temp.reshape(N_sweep, lam_plot.size, theta_plot.size, phi_plot.size)
    diff_CS = diff_CS_temp.reshape(N_sweep, lam_plot.size, theta_plot.size, phi_plot.size)
    
    r_save_proc = np.zeros((data_size[comm.rank], int(np.max(N_layer))))
    n_re_proc = np.zeros((data_size[comm.rank], lam_plot.size, int(np.max(N_layer))+1))
    n_im_proc = np.zeros((data_size[comm.rank], lam_plot.size, int(np.max(N_layer))+1))
    for nr in range(data_size[comm.rank]):
        r_save_proc[nr,:int(N_layer_proc[nr])] = radius_proc[nr]
        n_re_proc[nr,:,:int(N_layer_proc[nr])+1] = np.real(RI_proc[nr])
        n_im_proc[nr,:,:int(N_layer_proc[nr])+1] = np.imag(RI_proc[nr])
    
    data_size_temp = data_size*int(np.max(N_layer))
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    r_temp = np.zeros(N_sweep*int(np.max(N_layer)))
    comm.Gatherv(r_save_proc.reshape(-1), [r_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    r_save = r_temp.reshape(N_sweep, int(np.max(N_layer)))
    
    data_size_temp = data_size*lam_plot.size*(int(np.max(N_layer)) + 1)
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    n_re_temp = np.zeros(N_sweep*lam_plot.size*(int(np.max(N_layer)) + 1))
    n_im_temp = np.zeros(N_sweep*lam_plot.size*(int(np.max(N_layer)) + 1))
    comm.Gatherv(n_re_proc.reshape(-1), [n_re_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    comm.Gatherv(n_im_proc.reshape(-1), [n_im_temp, data_size_temp, data_disp_temp, MPI.DOUBLE], root=0)
    n_save = (n_re_temp + 1j*n_im_temp).reshape(N_sweep, lam_plot.size, int(np.max(N_layer))+1)
    
    if comm.rank == 0:
        # Remove Designs that are Too Small
        filter_mask = r_save[:,0] > d_low
        r_save = r_save[filter_mask,:]
        n_save = n_save[filter_mask,:]
        Q_sca = Q_sca[filter_mask,:]
        Q_abs = Q_abs[filter_mask,:]
        Q_ext = Q_ext[filter_mask,:]
        p = p[filter_mask,:,:,:]
        diff_CS = diff_CS[filter_mask,:,:,:]
        N_layer = N_layer[filter_mask]
        cost = cost[filter_mask]
        
        # Save Best Design
        cost_sort = np.argsort(cost)
        cost = cost[cost_sort]
        r_save = r_save[cost_sort,:][0,:]
        n_save = n_save[cost_sort,:,:][0,:,:]
        Q_sca = Q_sca[cost_sort,:][0,:]
        Q_abs = Q_abs[cost_sort,:][0,:]
        Q_ext = Q_ext[cost_sort,:][0,:]
        p = p[cost_sort,:,:,:][0,:,:,:]
        diff_CS = diff_CS[cost_sort,:,:,:][0,:,:,:]
        N_layer = N_layer[cost_sort][0]
            
        np.savez(output_filename, r=r_save, n=n_save,
                 Q_sca=Q_sca, Q_abs=Q_abs, Q_ext=Q_ext, p=p, diff_CS=diff_CS, N_layer=N_layer,
                 d_low=d_low, r_max=r_max, cost=cost)
        
        print('\n### Optimization Done\n', flush=True)