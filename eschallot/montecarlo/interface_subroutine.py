import numpy as np
from numba import jit
import eschallot.montecarlo.snell as snell

@jit(nopython=True, cache=True)
def interface_jit(n1, n2, downward, boundary, layer, layer_change,
                  photon_current_layer, photon_current_theta, photon_current_phi,
                  photon_interface_R, photon_interface_T, photon_TIR, photon_pol,
                  photon_current_pos, photon_prev_TIR_theta, photon_TIR_stop, photon_TIR_abs,
                  photon_exit, photon_theta_exit, photon_phi_exit, photon_x_exit, photon_y_exit, photon_z_exit):

    # Surface Normal Orientation (defined w.r.t. downward normal)
    surf_slope = 0
    th_surf = np.arctan(surf_slope) # note: th_surf < 0 for surf_slope < 0
    phi_surf = 2*np.pi*np.random.rand()
    
    # Vector Definitions
    v_inc = np.array([np.sin(photon_current_theta)*np.cos(photon_current_phi),
                      np.sin(photon_current_theta)*np.sin(photon_current_phi),
                      -np.cos(photon_current_theta)])
    v_norm = np.array([np.sin(th_surf)*np.cos(phi_surf),
                       np.sin(th_surf)*np.sin(phi_surf),
                       -np.cos(th_surf)])
    idotn = np.dot(v_inc, v_norm)

    # Angles of Incidence & Refraction
    th1 = np.arccos(np.abs(idotn))
    th2 = snell.snell(n1, n2, th1)

    # Reflection & Transmission Angles
    v_R = v_inc - 2*idotn*v_norm
    theta_R = np.arccos(-v_R[2])
    if downward:
        v_T = np.real(n1/n2)*v_inc + (np.cos(np.real(th2)) - np.real(n1/n2)*idotn)*v_norm
    else:
        v_T = np.real(n1/n2)*v_inc + (-np.cos(np.real(th2)) - np.real(n1/n2)*idotn)*v_norm
    theta_T = np.arccos(-v_T[2])
    
    if np.sin(theta_R) == 0:
        phi_R = 0
    else:
        phi_ratio = v_R[0]/np.sin(theta_R)
        if phi_ratio < -1:
            phi_ratio = -1
        elif phi_ratio > 1:
            phi_ratio = 1
        if v_R[1] >= 0:
            phi_R = np.arccos(phi_ratio)
        else:
            phi_R = 2*np.pi - np.arccos(phi_ratio)
    if np.sin(theta_T) == 0:
        phi_T = 0
    else:
        phi_ratio = v_T[0]/np.sin(theta_T)
        if phi_ratio < -1:
            phi_ratio = -1
        elif phi_ratio > 1:
            phi_ratio = 1
        if v_T[1] >= 0:
            phi_T = np.arccos(phi_ratio)
        else:
            phi_T = 2*np.pi - np.arccos(phi_ratio)
          
    if downward:
        if theta_R <= np.pi/2:
            theta_R = np.pi - theta_R
        if theta_T >= np.pi/2:
            theta_T = np.pi - theta_T
    else:
        if theta_R >= np.pi/2:
            theta_R = np.pi - theta_R
        if theta_T <= np.pi/2:
            theta_T = np.pi - theta_T

    if np.real((n1/n2)*np.sin(th1)) > 1:
        state = 'R'
        photon_TIR += 1
        photon_interface_R += 1
        photon_current_theta = theta_R
        photon_current_phi = phi_R

        if photon_prev_TIR_theta == th1: # TIR occurring repeatedly at the same angle (will continue ad infinitum if interfaces are flat)
            photon_TIR_stop += 1
        else:
            photon_TIR_stop = 0
        if photon_TIR_stop > 10:
            layer_change = 0
            photon_exit = 1
            photon_TIR_abs += 1 # treat it as a photon trapped in the film, which will eventually get absorbed
        photon_prev_TIR_theta = th1
    else:
        # Fresnel Reflectance
        r12_TE = (n1*np.cos(th1) - n2*np.cos(th2))/(n1*np.cos(th1) + n2*np.cos(th2))
        r12_TM = (n1/np.cos(th1) - n2/np.cos(th2))/(n1/np.cos(th1) + n2/np.cos(th2))
        if photon_interface_T == 0: #1st incidence on boundary (polarization is known)
            if photon_pol == 1: # Note: phi_inc = 0 always and phi is defined w.r.t. the x-axis; therefore, x-pol light will have E//plane(xz) and y-pol light will have E perp to plane(xz)
                r12 = r12_TE
            else:
                r12 = r12_TM
            R = np.abs(r12)**2
        else: #Otherwise, assume unpolarized incidence
            R = (np.abs(r12_TE)**2 + np.abs(r12_TM)**2)/2
    
        if np.random.rand() < R: #reflection
            state = 'R'
            photon_interface_R += 1
            photon_current_theta = theta_R
            photon_current_phi = phi_R
        else: #transmission
            state = 'T'
            photon_interface_T += 1
            photon_current_theta = theta_T
            photon_current_phi = phi_T
            if downward == 1:
                if photon_interface_T == 1:
                    photon_current_pos[2] = boundary[photon_current_layer] - 0.1
                photon_current_layer += 1
            else:
                if photon_interface_T == 1:
                    photon_current_pos[2] = boundary[photon_current_layer-1] + 0.1
                photon_current_layer -= 1
    
    if photon_interface_T == 0 and photon_interface_R == 1: # photon that is reflected off the very first interface
        if photon_current_layer == 0 or photon_current_layer == layer + 1:
            photon_exit = 1
            photon_theta_exit = photon_current_theta
            photon_phi_exit = photon_current_phi
            photon_x_exit, photon_y_exit, photon_z_exit = photon_current_pos
    
    return_tuple = (state, layer_change, photon_exit,
                    photon_interface_R, photon_interface_T, photon_TIR, photon_TIR_stop, photon_TIR_abs, photon_prev_TIR_theta,
                    photon_current_theta, photon_current_phi, photon_current_pos, photon_current_layer,
                    photon_theta_exit, photon_phi_exit, photon_x_exit, photon_y_exit, photon_z_exit)
    return return_tuple
