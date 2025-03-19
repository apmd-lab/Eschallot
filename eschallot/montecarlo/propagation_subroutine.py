import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def propagation_jit1(C_sca, C_abs, boundary, density, photon_index,
                     photon_interface_R, photon_interface_T, photon_particle_scat, photon_particle_abs,
                     photon_current_theta, photon_current_phi, photon_current_layer,
                     photon_next_pos):
    theta_temp = np.copy(np.array([photon_current_theta]))[0]
    phi_temp = np.copy(np.array([photon_current_phi]))[0]

    mfp_inv = (C_sca[photon_index] + C_abs[photon_index])*density

    if mfp_inv == 0:
        path_length = (boundary[photon_current_layer-1] - boundary[photon_current_layer])/np.abs(np.cos(theta_temp)) + 1
    else:
        mfp = 1/mfp_inv
        path_length = np.random.exponential(mfp)
    r = path_length*np.sin(theta_temp)
    
    dx = r*np.cos(phi_temp)
    dy = r*np.sin(phi_temp)
    dz = -path_length*np.cos(theta_temp)

    next_pos_save = np.copy(photon_next_pos)
    photon_next_pos = photon_next_pos + np.array([dx, dy, dz]).flatten()
    
    return (theta_temp, phi_temp, path_length, r, next_pos_save, photon_next_pos)

@jit(nopython=True, cache=True)
def propagation_jit2(layer, RI, photon_index, photon_wavelength,
                     photon_total_path, photon_next_pos, photon_track_pos,
                     photon_exit):
    arg = 0.0
    for l in range(layer):
        arg -= (((4*np.pi*np.imag(RI[l+1,photon_index]))/photon_wavelength)*photon_total_path[l])[0]
    photon_weight = np.exp(arg)
    
    photon_current_pos = np.copy(photon_next_pos)
    temp = np.copy(photon_track_pos)
    if temp.size == 3:
        photon_track_pos = np.concatenate((temp.reshape(3,1), photon_current_pos.reshape(3,1)), 1)
    else:
        photon_track_pos = np.concatenate((temp.reshape(3,-1), photon_current_pos.reshape(3,1)), 1)
    
    return (photon_weight, photon_exit, photon_current_pos, photon_track_pos)
