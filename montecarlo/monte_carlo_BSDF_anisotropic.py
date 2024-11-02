import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-11])

import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import montecarlo.snell as snell
import mie.simulate_particle as sim
import util.read_mat_data as rmd
import montecarlo.effective_medium_approximation as ema
import montecarlo.interface_subroutine as interf
import montecarlo.propagation_subroutine as propag
import time
from mpi4py import MPI
from numba import jit

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

class photon:
    def __init__(self, layer_number, wavelength, theta_inc, phi_inc, z_inc, polarization):
        self.weight = 1 #'percentage' of photon left after absorption by matrix
        self.index = np.random.randint(0, np.size(wavelength)) #index of the photon's wavelength
        self.wavelength = wavelength[self.index]
        if theta_inc == -1: #Random incidence
            self.theta_inc = np.pi/2*np.random.rand()
        else: #Incidence at specific angle
            self.theta_inc = theta_inc
        if phi_inc == -1: #Random incidence
            self.phi_inc = 2*np.pi*np.random.rand()
        else:
            self.phi_inc = phi_inc
        if polarization == 'random':
            self.pol = np.random.randint(0, 2) #Polarization (only for 1st interface); 0: TE, 1: TM
        elif polarization == 'x':
            self.pol = 0
        elif polarization == 'y':
            self.pol = 1
        self.x_inc = 0
        self.y_inc = 0
        
        self.current_theta = self.theta_inc # Note: all theta angles are defined with respect to a vector pointing down from the origin
        self.current_phi = self.phi_inc
        self.current_pos = np.array([0, 0, z_inc+0.1])
        self.next_pos = np.array([0, 0, z_inc-0.1])
        self.track_pos = self.current_pos.copy()
        self.current_layer = 0
        self.total_path = np.zeros((layer_number, 1))
        
        self.interface_R = 0 #number of interface reflections
        self.TIR = 0 #number of TIRs
        self.prev_TIR_theta = 0
        self.TIR_stop = 0
        self.interface_T = 0 #number of interface transmissions
        self.particle_scat = 0 #number of scattering by particle
        self.particle_abs = 0 #whether photon was absorbed by particle; 0: false, 1:true
        self.particle_PL = 0
        self.TIR_abs = 0
        self.scat_abs = 0
        self.PL_abs = 0
        self.exit = 0 #whether the photon escaped the film
        self.theta_exit = 0
        self.phi_exit = 0
        self.x_exit = 0
        self.y_exit = 0
        self.z_exit = 0

class monte_carlo:
    def __init__(self, wavelength, theta_inc, theta_out, phi_inc, phi_out, theta_in_BSDF, phi_in_BSDF, theta_out_BRDF_edge, theta_out_BRDF_center,
                 theta_out_BTDF_edge, theta_out_BTDF_center, phi_out_BSDF, wvl_for_polar_plots, angle_for_spectral_plots,
                 layer_thickness, RI, density, C_sca_surf, C_abs_surf, C_sca_bulk, C_abs_bulk, diff_scat_CS,
                 fine_roughness=0, coarse_roughness=0, antireflective=0, Lambertian_sub=0, perfect_absorber=0, isotropic=0, init_theta=0, init_phi=0, polarization='random'):
        """ wavelength: np.array of possible wavelengths (nm)
            theta_inc: np.array of polar angles of incidence used for the computation of the differential scattering cross section (radians)
            theta_out: np.array of polar scattering angles used for the computation of the differential scattering cross section (radians)
            phi_inc: np.array of azimuthal angles of incidence used for the computation of the differential scattering cross section (radians)
            phi_out: np.array of azimuthal scattering angles used for the computation of the differential scattering cross section (radians)
            theta_in_BSDF: np.array of polar angles of incidence on the film in the range [0, pi/2]
            phi_in_BSDF: np.array of azimuthal angles of incidence on the film in the range [0, 2*pi]
            theta_out_BRDF_edge: np.array of polar angles of exit on the side of incidence in the range [pi/2, pi]
            theta_out_BRDF_center: np.array of polar angles at the center of each segment defined by theta_out_BRDF_edge
            theta_out_BTDF_edge: np.array of polar angles of exit on the transmission side in the range [0, pi/2]
            theta_out_BTDF_center: np.array of polar angles at the center of each segment defined by theta_out_BTDF_edge
            phi_out_BSDF: np.array of azimuthal angles of exit
            wvl_for_polar_plots: np.array of wavelengths at which to plot the angular distribution of reflected/transmitted photons
            angle_for_spectral_plots: np.array of angles (N x 2) at which to plot the spectral distribution of reflected/transmitted photons
            layer_thickness: np.array of thicknesses of each layer from top to bottom (nm)
            RI: refractive index from top to bottom (layer + 2) x (wavelength)
            density: particle density (number per nm^3)
                     if isotropic, (layer) x (particle type) x (particle radius)
                     if anisotropic, (layer) x (particle type)
            C_sca_surf: particle scattering cross section on surface (particle type) x (particle radius) x (wavelength)
                        if anisotropic, (particle type) x (polarization) x (wavelength) x (incident polar angle) x (incident azimuthal angle)
            C_abs_surf: particle absorption cross section on surface (particle type) x (particle radius) x (wavelength)
                        if anisotropic, (particle type) x (polarization) x (wavelength) x (incident polar angle) x (incident azimuthal angle)
            C_sca_bulk: particle scattering cross section in each layer (layer) x (particle type) x (particle radius) x (wavelength)
                        if anisotropic, (layer) x (particle type) x (polarization) x (wavelength) x (incident polar angle) x (incident azimuthal angle)
            C_abs_bulk: particle absorption cross section in each layer (layer) x (particle type) x (particle radius) x (wavelength)
                        if anisotropic, (layer) x (particle type) x (polarization) x (wavelength) x (incident polar angle) x (incident azimuthal angle)
            diff_scat_CS: differential scattering cross section; i.e. phase function x structure factor
                          if isotropic, (layer) x (particle type) x (particle radius) x (polarization) x (wavelength) x (scattered polar angle) x (scattererd azimuthal angle)
                          if anisotropic, (layer) x (particle type) x (polarization) x (wavelength)
                                          x (incident polar angle) x (incident azimuthal angle) x (scattered polar angle) x (scattered azimuthal angle)
            fine_roughness: fraction of surface area covered by particle-scale roughness; 0~1
            coarse_roughness: standard deviation of the distribution of surface slopes (mean = 0)
            antireflective: assume antireflective coating on the surface of incidence
            Lambertian_sub: assume a Lambertian reflector as the substrate
            perfect_absorber: assume the substrate is perfectly absorbing
            isotropic: particle scattering response is independent of the angle of incidence (ex. particle is spherical)
            init_theta: polar angle of incidence (-1 if random incidence) (radians)
            init_phi: azimuthal angle of incidence (-1 if random incidence) (radians)
            polarization: polarization of incident light ('x', 'y', or 'random') """
        
        self.wavelength = wavelength
        self.theta_inc = theta_inc
        self.theta_out = theta_out
        self.phi_inc = phi_inc
        self.phi_out = phi_out
        self.theta_in_BSDF = theta_in_BSDF
        self.phi_in_BSDF = phi_in_BSDF
        self.theta_out_BRDF_edge = theta_out_BRDF_edge
        self.theta_out_BRDF_center = theta_out_BRDF_center
        self.theta_out_BTDF_edge = theta_out_BTDF_edge
        self.theta_out_BTDF_center = theta_out_BTDF_center
        self.phi_out_BSDF = phi_out_BSDF
        self.wvl_for_polar_plots = wvl_for_polar_plots
        self.angle_for_spectral_plots = angle_for_spectral_plots
        if isotropic:
            self.layer, self.n_type, self.n_particle, self.n_pol, self.wvl, self.th_out, self.ph_out = np.shape(diff_scat_CS)
        else:
            self.layer, self.n_type, self.n_particle, self.n_pol, self.wvl, self.th_in, self.ph_in, self.th_out, self.ph_out = np.shape(diff_scat_CS)
        
        #Location of each boundary (bottommost boundary is at z = 0)
        self.boundary = np.sum(layer_thickness)*np.ones(self.layer+1)
        for l in range(1, self.layer+1):
            self.boundary[l] = self.boundary[l-1] - layer_thickness[l-1]
            
        self.RI = RI
        self.density = density
        self.C_sca_surf = C_sca_surf
        self.C_abs_surf = C_abs_surf
        self.C_sca_bulk = C_sca_bulk
        self.C_abs_bulk = C_abs_bulk
        
        #Normalization of the differential scattering cross section
        if isotropic:
            diff_scat_CS /= np.sum(diff_scat_CS, axis=(5,6))[:,:,:,:,:,np.newaxis,np.newaxis]
        else:
            diff_scat_CS /= np.sum(diff_scat_CS, axis=(6,7))[:,:,:,:,:,:,np.newaxis,np.newaxis]
        self.diff_scat_CS = diff_scat_CS
        
        self.fine_roughness = fine_roughness
        self.coarse_roughness = coarse_roughness
        self.antireflective = antireflective
        self.Lambertian_sub = Lambertian_sub
        self.perfect_absorber = perfect_absorber
        self.isotropic = isotropic
        self.init_theta = init_theta
        self.init_phi = init_phi
        self.polarization = polarization
    
    def interface(self, photon):
        """ this function only updates the photon's propagation angle
            photon position is unchanged (except for a minute shift along z) """
        n1 = self.RI[photon.current_layer, photon.index]
        if photon.current_theta <= np.pi/2: # downward moving photon
            n2 = self.RI[photon.current_layer+1, photon.index]
            self.downward = 1
        else:
            n2 = self.RI[photon.current_layer-1, photon.index]
            self.downward = 0    
        
        if self.downward and photon.current_layer == self.layer and self.Lambertian_sub:
            # above conditions check if the photon reached the lower interface of the bottommost layer
            state = 'R'
            photon.interface_R += 1
            
            upward_theta = self.theta[self.theta > np.pi/2]
            photon.current_theta = np.random.choice(upward_theta,
                                                    p=np.cos(upward_theta)/np.sum(np.cos(upward_theta)))
            photon.current_phi = 2*np.pi*np.random.rand()
            photon.current_pos[2] += 0.1
        elif self.downward and photon.current_layer == self.layer and self.perfect_absorber:
            state = 'A'
            photon.weight = 0
            photon.current_pos[2] = -0.1
        else:
            (state, self.layer_change, photon.exit,
             photon.interface_R, photon.interface_T, photon.TIR, photon.TIR_stop, photon.TIR_abs, photon.prev_TIR_theta,
             photon.current_theta, photon.current_phi, photon.current_pos, photon.current_layer,
             photon.theta_exit, photon.phi_exit,
             photon.x_exit, photon.y_exit, photon.z_exit) = interf.interface_jit(n1, n2, self.coarse_roughness, self.downward, self.boundary, self.layer, self.layer_change,
                                                                                 photon.current_layer, photon.current_theta, photon.current_phi,
                                                                                 photon.interface_R, photon.interface_T, photon.TIR, photon.pol,
                                                                                 photon.current_pos, photon.prev_TIR_theta, photon.TIR_stop, photon.TIR_abs,
                                                                                 photon.exit, photon.theta_exit, photon.phi_exit, photon.x_exit, photon.y_exit, photon.z_exit)
                                                       
        return state
    
    def propagation(self, photon):
        th_in = np.argmin(np.abs(photon.current_theta - self.theta_inc))
        ph_in = np.argmin(np.abs(photon.current_phi - self.phi_inc))
        if self.isotropic:
            C_sca_bulk_temp = self.C_sca_bulk.copy()
            C_abs_bulk_temp = self.C_abs_bulk.copy()
            C_sca_surf_temp = self.C_sca_surf.copy()
            C_abs_surf_temp = self.C_abs_surf.copy()
        else:
            C_sca_bulk_temp = self.C_sca_bulk[:,:,photon.pol,:,th_in,ph_in]
            C_abs_bulk_temp = self.C_abs_bulk[:,:,photon.pol,:,th_in,ph_in]
            C_sca_surf_temp = self.C_sca_surf[:,photon.pol,:,th_in,ph_in]
            C_abs_surf_temp = self.C_abs_surf[:,photon.pol,:,th_in,ph_in]
        (theta_temp, phi_temp, path_length,
         r, next_pos_save, photon.next_pos) = propag.propagation_jit1(self.fine_roughness, C_sca_bulk_temp, C_abs_bulk_temp, C_sca_surf_temp, C_abs_surf_temp,
                                                                      self.boundary, self.density, photon.index,
                                                                      photon.interface_R, photon.interface_T, photon.particle_scat, photon.particle_abs,
                                                                      photon.current_theta, photon.current_phi, photon.current_layer,
                                                                      photon.next_pos, self.isotropic)
        
        if self.layer_crossing(photon):
            self.layer_change = 1
            state = self.interface(photon)
            if state == 'A': # absorption by a perfectly absorbing substrate
                self.layer_change = 0
                photon.exit = 1
                photon.theta_exit = photon.current_theta
                photon.phi_exit = photon.current_phi
                photon.x_exit, photon.y_exit, photon.z_exit = photon.next_pos
            else:
                if state == 'T':
                    if self.downward:
                        lyr1 = photon.current_layer - 1
                        offset = -0.1
                    else:
                        lyr1 = photon.current_layer + 1
                        offset = 0.1
                else:
                    lyr1 = photon.current_layer
                    if self.downward:
                        offset = 0.1
                    else:
                        offset = -0.1
                
                if self.downward:
                    r_new = (photon.current_pos[2] - self.boundary[lyr1])/(photon.current_pos[2] - photon.next_pos[2])*r
                    path_length_new = (photon.current_pos[2] - self.boundary[lyr1])/(photon.current_pos[2] - photon.next_pos[2])*path_length
                else:
                    r_new = (self.boundary[lyr1-1] - photon.current_pos[2])/(photon.next_pos[2] - photon.current_pos[2])*r
                    path_length_new = (self.boundary[lyr1-1] - photon.current_pos[2])/(photon.next_pos[2] - photon.current_pos[2])*path_length
                
                photon.next_pos = next_pos_save
    
                dx = r_new*np.cos(phi_temp)
                dy = r_new*np.sin(phi_temp)
                dz = -path_length_new*np.cos(theta_temp) + offset
                
                photon.next_pos = photon.next_pos + np.array([dx, dy, dz]).flatten()
                photon.total_path[lyr1-1] = photon.total_path[lyr1-1] + path_length_new
                
                if photon.current_layer == 0 or photon.current_layer == self.layer + 1:
                    # after interface update, the photon is outside the film
                    self.layer_change = 0
                    photon.exit = 1
                    photon.theta_exit = photon.current_theta
                    photon.phi_exit = photon.current_phi
                    photon.x_exit, photon.y_exit, photon.z_exit = photon.next_pos
                    # if there is a Lambertian substrate, the interface function will catch all photons passing out the bottommost
                    # interface and there should be no transmission that would cause current_layer == layer + 1
                
                (photon.weight, photon.exit,
                 photon.current_pos, photon.track_pos) = propag.propagation_jit2(self.layer, self.RI, photon.index, photon.wavelength,
                                                                                 photon.total_path, photon.next_pos, photon.track_pos,
                                                                                 photon.exit)
        else:
            self.layer_change = 0
            photon.total_path[photon.current_layer-1] = photon.total_path[photon.current_layer-1] + path_length
        
            (photon.weight, photon.exit,
             photon.current_pos, photon.track_pos) = propag.propagation_jit2(self.layer, self.RI, photon.index, photon.wavelength,
                                                                             photon.total_path, photon.next_pos, photon.track_pos,
                                                                             photon.exit)
               
    def particle(self, photon):
        # note that the surface cross sections are only used for the initial mfp (not for actual cross sections)
        if self.isotropic:
            th_in = 0 # differential cross section was computed assuming theta=0, phi=0 incidence
            ph_in = 0
            C_sca = self.C_sca_bulk[photon.current_layer-1,:,:,photon.index]
            C_abs = self.C_abs_bulk[photon.current_layer-1,:,:,photon.index]

            mfp_inv = (C_sca + C_abs)*self.density[photon.current_layer-1,:,:]

            particle_type = int(np.random.choice(np.linspace(0, self.n_type, self.n_type, endpoint=False),
                                                 p=np.sum(mfp_inv, axis=1)/np.sum(mfp_inv)))
            particle_radius = int(np.random.choice(np.linspace(0, self.n_particle, self.n_particle, endpoint=False),
                                                   p=mfp_inv[particle_type,:]/np.sum(mfp_inv[particle_type,:])))
            prob = (C_abs[particle_type,particle_radius])/(C_abs[particle_type,particle_radius] + C_sca[particle_type,particle_radius])
        else:
            th_in = np.argmin(np.abs(photon.current_theta - self.theta_inc))
            ph_in = np.argmin(np.abs(photon.current_phi - self.phi_inc))
            C_sca = self.C_sca_bulk[photon.current_layer-1,:,photon.pol,photon.index,th_in,ph_in]
            C_abs = self.C_abs_bulk[photon.current_layer-1,:,photon.pol,photon.index,th_in,ph_in]
            
            mfp_inv = (C_sca + C_abs)*self.density[photon.current_layer-1,:]
            
            particle_type = int(np.random.choice(np.linspace(0, self.n_particle, self.n_particle, endpoint=False),
                                                 p=mfp_inv/np.sum(mfp_inv))) # probability should take into account both cross sections and density
            prob = (C_abs[particle_type])/(C_abs[particle_type] + C_sca[particle_type])
        
        if np.random.rand() < prob: # photon is absorbed
            photon.particle_abs = 1
            photon.exit = 1
        else: # photon is scattered
            photon.particle_scat += 1
            if photon.particle_scat > 1e4:
                photon.exit = 1 # early stop condition for photons that scatter too much
                photon.scat_abs = 1
                
                layer_thickness = self.boundary[:-1] - self.boundary[1:]
                if self.isotropic:
                    decay_coeff = np.sum(self.density[:photon.current_layer-1,:,:]*self.C_sca_bulk[:photon.current_layer-1,:,:,photon.index]\
                                  *layer_thickness[:photon.current_layer-1][:,np.newaxis,np.newaxis])
                    decay_coeff += np.sum(self.density[photon.current_layer-1,:,:]*self.C_sca_bulk[photon.current_layer-1,:,:,photon.index]\
                                   *np.abs(self.boundary[photon.current_layer-1] - photon.current_pos[2]))
                    R = np.exp(-decay_coeff)
                    
                    decay_coeff = np.sum(self.density[photon.current_layer:,:,:]*self.C_sca_bulk[photon.current_layer:,:,:,photon.index]\
                                  *layer_thickness[photon.current_layer:][:,np.newaxis,np.newaxis])
                    decay_coeff += np.sum(self.density[photon.current_layer-1,:,:]*self.C_sca_bulk[photon.current_layer-1,:,:,photon.index]\
                                   *np.abs(self.boundary[photon.current_layer] - photon.current_pos[2]))
                    T = np.exp(-decay_coeff)
                    
                    photon.pR = R/(R + T)
                else:
                    decay_coeff = np.sum(self.density[:photon.current_layer-1,:]*self.C_sca_bulk[:photon.current_layer-1,:,photon.pol,photon.index,th_in,ph_in]\
                                  *layer_thickness[:photon.current_layer-1][:,np.newaxis,np.newaxis])
                    decay_coeff += np.sum(self.density[photon.current_layer-1,:]*self.C_sca_bulk[photon.current_layer-1,:,photon.pol,photon.index,th_in,ph_in]\
                                   *np.abs(self.boundary[photon.current_layer-1] - photon.current_pos[2]))
                    R = np.exp(-decay_coeff)
                    
                    decay_coeff = np.sum(self.density[photon.current_layer:,:]*self.C_sca_bulk[photon.current_layer:,:,photon.pol,photon.index,th_in,ph_in]\
                                  *layer_thickness[photon.current_layer:][:,np.newaxis,np.newaxis])
                    decay_coeff += np.sum(self.density[photon.current_layer-1,:]*self.C_sca_bulk[photon.current_layer-1,:,photon.pol,photon.index,th_in,ph_in]\
                                   *np.abs(self.boundary[photon.current_layer] - photon.current_pos[2]))
                    T = np.exp(-decay_coeff)
                    
                    photon.pR = R/(R + T)
                    
            else:
                if self.isotropic:
                    diff_scat_CS_flat = self.diff_scat_CS[photon.current_layer-1,particle_type,particle_radius,photon.pol,photon.index,:,:].reshape(-1)
                else:
                    diff_scat_CS_flat = self.diff_scat_CS[photon.current_layer-1,particle_type,photon.pol,photon.index,th_in,ph_in,:,:].reshape(-1)
                ind_choice = np.random.choice(np.linspace(0, self.th_out*self.ph_out, self.th_out*self.ph_out, endpoint=False),
                                              p=diff_scat_CS_flat)
                
                theta_temp = self.theta_out[int(ind_choice//self.ph_out)]
                phi_temp = self.phi_out[int(ind_choice%self.ph_out)]
                theta_rot = photon.current_theta - self.theta_inc[th_in] # needed due to limited number of incidence angle sampling when computing cross sections
                phi_rot = photon.current_phi - self.phi_inc[ph_in]       # note: here, we're trying to rotate the axes from photon.current_theta to self.theta_inc[th_in]
        
                v_temp = np.array([np.sin(theta_temp)*np.cos(phi_temp),
                                   np.sin(theta_temp)*np.sin(phi_temp),
                                   -np.cos(theta_temp)])
                Rot_y = np.array([[np.cos(-theta_rot),0,np.sin(-theta_rot)], # -theta_rot because here, theta is defined w.r.t. the -z axis
                                  [0,1,0],
                                  [-np.sin(-theta_rot),0,np.cos(-theta_rot)]])
                Rot_z = np.array([[np.cos(phi_rot),-np.sin(phi_rot),0],
                                  [np.sin(phi_rot),np.cos(phi_rot),0],
                                  [0,0,1]])
                v_sca = Rot_z @ Rot_y @ v_temp
                
                if np.abs(v_sca[2]) > 1 and np.abs(v_sca[2]) < 1 + 1e-3:
                    theta_new = np.arccos(-np.sign(v_sca[2]))
                else:
                    theta_new = np.arccos(-v_sca[2])
                    
                if theta_new == 0:
                    photon.current_phi = 0
                else:
                    phi_temp = v_sca[0]/np.sin(theta_new)
                    if phi_temp < -1:
                        phi_temp = -1
                    elif phi_temp > 1:
                        phi_temp = 1
                    if v_sca[1] >= 0:
                        photon.current_phi = np.arccos(phi_temp)
                    else:
                        photon.current_phi = 2*np.pi - np.arccos(phi_temp)
                        
                photon.current_theta = theta_new
        
    def layer_crossing(self, photon):
        next_layer = 0
        for l in range(1, self.layer+1):
            if photon.next_pos[2] > self.boundary[l] and photon.next_pos[2] < self.boundary[l-1]:
                next_layer = l
                break
            
        return next_layer != photon.current_layer
    
    def normal_hemispherical(self, directory_output, comm, size, rank, status, num_photons, identifier, subgroup):
        """ Formatting of photon_state:
            index   0                       1                     2                                    3
            value   initial index of wvl    final index of wvl    state of photon                      weight of photon
                                                                  (0:R, 1:T, 2:A_particle, 3:A_TIR,
                                                                   4:R_scat, 5:T_scat, 6:A_PL, 7:R_PL,
                                                                   8:T_PL)
            
                    4                          5                              6                     7                          8
                    polar angle of incidence   azimuthal angle of incidence   polar angle of exit   azimuthal angle of exit    scattering count
        """
        if rank == 0:
            quo, rem = divmod(num_photons, int(size/subgroup))
            data_size = [quo + 1 if p < rem else quo for p in range(int(size/subgroup))]
            data_size = np.array(data_size).astype(np.int64)
            data_disp = [sum(data_size[:p]) for p in range(int(size/subgroup))]
            data_disp = np.array(data_disp).astype(np.int64)
        else:
            data_size = np.zeros(int(size/subgroup), dtype=np.int64)
            data_disp = np.zeros(int(size/subgroup), dtype=np.int64)

        comm.Bcast(data_size, root=0)
        comm.Bcast(data_disp, root=0)

        R_spec = np.zeros(self.wvl)
        R_diff = np.zeros(self.wvl)
        R_scat = np.zeros(self.wvl)
        I = np.zeros(self.wvl)
        A_medium = np.zeros(self.wvl)
        A_particle = np.zeros(self.wvl)
        A_TIR = np.zeros(self.wvl)
        T_ball = np.zeros(self.wvl) # ballistic
        T_diff = np.zeros(self.wvl)
        T_scat = np.zeros(self.wvl)
        Ns = np.zeros(self.wvl)
    
        inc_angle = np.zeros((self.theta_in_BSDF.size, self.phi_in_BSDF.size, self.wvl))
        reflect_angle_spec = np.zeros((self.theta_out_BRDF_center.size, self.phi_out_BSDF.size, self.wvl))
        reflect_angle_diff = np.zeros((self.theta_out_BRDF_center.size, self.phi_out_BSDF.size, self.wvl))
        transmit_angle_ball = np.zeros((self.theta_out_BTDF_center.size, self.phi_out_BSDF.size, self.wvl))
        transmit_angle_diff = np.zeros((self.theta_out_BTDF_center.size, self.phi_out_BSDF.size, self.wvl))

        if int(rank % subgroup) == 0: #only on head processes
            with open(directory_output + "/logs/" + identifier + "_MC_log_head" + str(rank) + ".txt", 'w') as f:
                f.write("Monte Carlo Simulation (MPI_ver) Log File -- Head Process\n")
                f.write("Incidence Angle: theta = %f, phi = %f\n" %(self.init_theta*180/np.pi, self.init_phi*180/np.pi))

            count = 0
            num = data_disp[rank//subgroup]
            if rank//subgroup + 1 == np.size(data_disp):
                num_end = num_photons
            else:
                num_end = data_disp[rank//subgroup+1]
            cmd = 1
            while num < num_end and num < data_disp[rank//subgroup] + subgroup - 1: #initial distribution of work
                count += 1
                num += 1 #this is only for generating the tags

                comm.send(cmd, dest=(rank//subgroup)*subgroup+count, tag=num%1e4)
            
            while True:
                photon_state = np.zeros(9).astype(np.float64)
                req = comm.Irecv(photon_state, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                req.wait(status)

                num += 1
                if num >= num_end:
                    break
                
                quo, rem = divmod(num-data_disp[rank//subgroup], int((num_end-data_disp[rank//subgroup])/100))
                if rem == 0:
                    with open(directory_output + "/logs/" + identifier + "_MC_log_head" + str(rank) + ".txt", 'a') as f:
                        f.write("%f percent complete\n" % (quo))
                    
                    N_plots = 6 + self.wvl_for_polar_plots.size + self.angle_for_spectral_plots.size
                    rank_for_plots = np.zeros(N_plots)
                    multiplier = 0
                    for n_plots in range(N_plots):
                        if multiplier*subgroup >= size:
                            multiplier = 0
                        rank_for_plots[n_plots] = multiplier*subgroup
                        multiplier += 1
                        
                    if rank == rank_for_plots[0] and self.wavelength.size > 1:
                        # Reflectance Spectrum
                        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
                        ax.plot(self.wavelength, (R_spec+R_diff+R_scat)/I, linewidth=1, color='darkblue', label='R_tot')
                        ax.plot(self.wavelength, R_spec/I, linewidth=1, linestyle='dashed', color='blue', label='R_spec')
                        ax.plot(self.wavelength, R_diff/I, linewidth=1, linestyle='dashed', color='slateblue', label='R_diff')
                        ax.plot(self.wavelength, R_scat/I, linewidth=1, linestyle='dashed', color='cyan', label='R_scat')
                        ax.set_ylim(-0.01, 1.01)
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Reflectance')
                        ax.legend()
                        plt.savefig(directory_output + '/plots/' + identifier + '_R_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    if rank == rank_for_plots[1] and self.wavelength.size > 1:
                        # Transmittance Spectrum
                        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
                        ax.plot(self.wavelength, (T_ball+T_diff+T_scat)/I, linewidth=1, color='darkred', label='T_tot')
                        ax.plot(self.wavelength, T_ball/I, linewidth=1, linestyle='dashed', color='red', label='T_ballistic')
                        ax.plot(self.wavelength, T_diff/I, linewidth=1, linestyle='dashed', color='lightcoral', label='T_diff')
                        ax.plot(self.wavelength, T_scat/I, linewidth=1, linestyle='dashed', color='firebrick', label='T_scat')
                        ax.set_ylim(-0.01, 1.01)
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Transmittance')
                        ax.legend()
                        plt.savefig(directory_output + '/plots/' + identifier + '_T_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    if rank == rank_for_plots[2] and self.wavelength.size > 1:
                        # Absorption Spectrum
                        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
                        ax.plot(self.wavelength, (A_medium+A_particle+A_TIR)/I, linewidth=1, color='goldenrod', label='A_total')
                        ax.plot(self.wavelength, A_medium/I, linewidth=1, linestyle='dashed', color='orange', label='A_medium')
                        ax.plot(self.wavelength, A_particle/I, linewidth=1, linestyle='dashed', color='gold', label='A_particle')
                        ax.plot(self.wavelength, A_TIR/I, linewidth=1, linestyle='dashed', color='olive', label='A_TIR')
                        ax.set_ylim(-0.01, 1.01)
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Absorption')
                        ax.legend()
                        plt.savefig(directory_output + '/plots/' + identifier + '_A_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    if rank == rank_for_plots[3] and self.wavelength.size > 1:
                        # Average Number of Scattering Events (spectrum)
                        fig, ax = plt.subplots(figsize=[12,5], dpi=100)
                        ax.plot(self.wavelength, Ns/I, linewidth=1, color='teal', label='N_scatter')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.legend()
                        plt.savefig(directory_output + '/plots/' + identifier + '_N_scatter_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    if rank == rank_for_plots[4] and self.wavelength.size > 1:
                        # Polar Angle vs. Spectrum Plots for Photons that Didn't Scatter
                        fig, ax = plt.subplots(1, 2, figsize=[12,5], dpi=100)
                        xgrid_R, ygrid_R = np.meshgrid(self.wavelength, self.theta_out_BRDF_center*180/np.pi, indexing='ij')
                        xgrid_T, ygrid_T = np.meshgrid(self.wavelength, self.theta_out_BTDF_center*180/np.pi, indexing='ij')
                        vmax = np.nanmax(np.sum(reflect_angle_spec, axis=1)/I[np.newaxis,:])
                        im1 = ax[0].contourf(xgrid_R, ygrid_R, (np.sum(reflect_angle_spec, axis=1)/I[np.newaxis,:]).T, cmap='plasma', vmax=vmax, vmin=0, levels=100)
                        vmax = np.nanmax(np.sum(transmit_angle_ball, axis=1)/I[np.newaxis,:])
                        im2 = ax[1].contourf(xgrid_T, ygrid_T, (np.sum(transmit_angle_ball, axis=1)/I[np.newaxis,:]).T, cmap='plasma', vmax=vmax, vmin=0, levels=100)
                        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.85, wspace=0.15, hspace=0.2)
                        c_ax = fig.add_axes([0.91, 0.05, 0.015, 0.9])
                        fig.colorbar(im1, cax=c_ax)
                        ax[0].set_xlim(self.wavelength[0], self.wavelength[-1])
                        ax[1].set_xlim(self.wavelength[0], self.wavelength[-1])
                        ax[0].set_ylim(90, 180)
                        ax[1].set_ylim(0, 90)
                        ax[0].set_xlabel('Wavelength (nm)')
                        ax[1].set_xlabel('Wavelength (nm)')
                        ax[0].set_ylabel('Polar Exit Angle (deg.)')
                        ax[0].set_title('Specular Reflection')
                        ax[1].set_title('Ballistic Transmission')
                        plt.savefig(directory_output + '/plots/' + identifier + '_Angular_specR_ballT_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    if rank == rank_for_plots[5] and self.wavelength.size > 1:
                        # Polar Angle vs. Spectrum Plots for Photons that Scattered at least Once
                        fig, ax = plt.subplots(1, 2, figsize=[12,5], dpi=100)
                        xgrid_R, ygrid_R = np.meshgrid(self.wavelength, self.theta_out_BRDF_center*180/np.pi, indexing='ij')
                        xgrid_T, ygrid_T = np.meshgrid(self.wavelength, self.theta_out_BTDF_center*180/np.pi, indexing='ij')
                        vmax = np.nanmax(np.sum(reflect_angle_diff, axis=1)/I[np.newaxis,:])
                        im1 = ax[0].contourf(xgrid_R, ygrid_R, (np.sum(reflect_angle_diff, axis=1)/I[np.newaxis,:]).T, cmap='plasma', vmax=vmax, vmin=0, levels=100)
                        vmax = np.nanmax(np.sum(transmit_angle_diff, axis=1)/I[np.newaxis,:])
                        im2 = ax[1].contourf(xgrid_T, ygrid_T, (np.sum(transmit_angle_diff, axis=1)/I[np.newaxis,:]).T, cmap='plasma', vmax=vmax, vmin=0, levels=100)
                        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.85, wspace=0.15, hspace=0.2)
                        c_ax = fig.add_axes([0.91, 0.05, 0.015, 0.9])
                        fig.colorbar(im1, cax=c_ax)
                        ax[0].set_xlim(self.wavelength[0], self.wavelength[-1])
                        ax[1].set_xlim(self.wavelength[0], self.wavelength[-1])
                        ax[0].set_ylim(90, 180)
                        ax[1].set_ylim(0, 90)
                        ax[0].set_xlabel('Wavelength (nm)')
                        ax[1].set_xlabel('Wavelength (nm)')
                        ax[0].set_ylabel('Polar Exit Angle (deg.)')
                        ax[0].set_title('Diffuse Reflection')
                        ax[1].set_title('Diffuse Transmission')
                        plt.savefig(directory_output + '/plots/' + identifier + '_Angular_diffRT_inc_angle_' + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                        plt.close()
                    
                    # Full Angle Distribution at Specific Wavelengths
                    for n_wvl in range(self.wvl_for_polar_plots.size):
                        if rank == rank_for_plots[5+n_wvl]:
                            # Photons that Didn't Scatter
                            wvl_index = np.argmin(np.abs(self.wavelength - self.wvl_for_polar_plots[n_wvl]))
                            fig, ax = plt.subplots(1, 2, figsize=[12,6], dpi=100, subplot_kw=dict(projection='polar'))
                            phi_plot = np.append(self.phi_out_BSDF, 2*np.pi)
                            thgrid_R, rgrid_R = np.meshgrid(phi_plot, (np.pi-self.theta_out_BRDF_center)*180/np.pi, indexing='ij')
                            reflect_plot = np.zeros((self.theta_out_BRDF_center.size, self.phi_out_BSDF.size+1))
                            reflect_plot[:,:-1] = reflect_angle_spec[:,:,wvl_index]
                            reflect_plot[:,-1] = reflect_angle_spec[:,0,wvl_index]
                            vmax = np.nanmax(reflect_plot/I[wvl_index])
                            im1 = ax[0].contourf(thgrid_R, rgrid_R, reflect_plot.T/I[wvl_index], cmap='plasma', vmax=vmax, vmin=0, levels=100)
                            thgrid_T, rgrid_T = np.meshgrid(phi_plot, self.theta_out_BTDF_center*180/np.pi, indexing='ij')
                            transmit_plot = np.zeros((self.theta_out_BTDF_center.size, self.phi_out_BSDF.size+1))
                            transmit_plot[:,:-1] = transmit_angle_ball[:,:,wvl_index]
                            transmit_plot[:,-1] = transmit_angle_ball[:,0,wvl_index]
                            vmax = np.nanmax(transmit_plot/I[wvl_index])
                            im2 = ax[1].contourf(thgrid_T, rgrid_T, transmit_plot.T/I[wvl_index], cmap='plasma', vmax=vmax, vmin=0, levels=100)
                            fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.85, wspace=0.15, hspace=0.2)
                            c_ax = fig.add_axes([0.91, 0.05, 0.015, 0.9])
                            fig.colorbar(im1, cax=c_ax)
                            ax[0].set_ylim(0, 90)
                            ax[1].set_ylim(0, 90)
                            ax[0].set_title('Specular Reflection @ ' + str(np.round(self.wavelength[wvl_index])))
                            ax[1].set_title('Ballistic Transmission @ ' + str(np.round(self.wavelength[wvl_index])))
                            plt.savefig(directory_output + '/plots/' + identifier + '_Exit_angle_distribution_wvl' + str(int(self.wavelength[wvl_index])) + '_spec_ball_inc_angle_'\
                                        + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                            plt.close()
                    
                            # Scattered Photons
                            fig, ax = plt.subplots(1, 2, figsize=[12,6], dpi=100, subplot_kw=dict(projection='polar'))
                            phi_plot = np.append(self.phi_out_BSDF, 2*np.pi)
                            thgrid_R, rgrid_R = np.meshgrid(phi_plot, (np.pi-self.theta_out_BRDF_center)*180/np.pi, indexing='ij')
                            reflect_plot = np.zeros((self.theta_out_BRDF_center.size, self.phi_out_BSDF.size+1))
                            reflect_plot[:,:-1] = reflect_angle_diff[:,:,wvl_index]
                            reflect_plot[:,-1] = reflect_angle_diff[:,0,wvl_index]
                            vmax = np.nanmax(reflect_plot/I[wvl_index])
                            im1 = ax[0].contourf(thgrid_R, rgrid_R, reflect_plot.T/I[wvl_index], cmap='plasma', vmax=vmax, vmin=0, levels=100)
                            thgrid_T, rgrid_T = np.meshgrid(phi_plot, self.theta_out_BTDF_center*180/np.pi, indexing='ij')
                            transmit_plot = np.zeros((self.theta_out_BTDF_center.size, self.phi_out_BSDF.size+1))
                            transmit_plot[:,:-1] = transmit_angle_diff[:,:,wvl_index]
                            transmit_plot[:,-1] = transmit_angle_diff[:,0,wvl_index]
                            vmax = np.nanmax(transmit_plot/I[wvl_index])
                            im2 = ax[1].contourf(thgrid_T, rgrid_T, transmit_plot.T/I[wvl_index], cmap='plasma', vmax=vmax, vmin=0, levels=100)
                            fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.85, wspace=0.15, hspace=0.2)
                            c_ax = fig.add_axes([0.91, 0.05, 0.015, 0.9])
                            fig.colorbar(im1, cax=c_ax)
                            ax[0].set_ylim(0, 90)
                            ax[1].set_ylim(0, 90)
                            ax[0].set_title('Diffuse Reflection @ ' + str(np.round(self.wavelength[wvl_index])))
                            ax[1].set_title('Diffuse Transmission @ ' + str(np.round(self.wavelength[wvl_index])))
                            plt.savefig(directory_output + '/plots/' + identifier + '_Exit_angle_distribution_wvl' + str(int(self.wavelength[wvl_index])) + '_diff_inc_angle_'\
                                        + str(int(self.init_theta*180/np.pi)) + '_' + str(int(self.init_phi*180/np.pi)))
                            plt.close()
                    
                    # Spectra at Specific Angles
                    for n_angle in range(self.angle_for_spectral_plots.shape[0]):
                        if rank == rank_for_plots[5+self.wvl_for_polar_plots.size+n_angle] and self.wavelength.size > 1:
                            fig, ax = plt.subplots(figsize=[12,5], dpi=100)
                            ph_index = np.argmin(np.abs(self.phi_out_BSDF - self.angle_for_spectral_plots[n_angle,1]))
                            if self.angle_for_spectral_plots[n_angle,0] < np.pi/2:
                                th_index = np.argmin(np.abs(self.theta_out_BTDF_center - self.angle_for_spectral_plots[n_angle,0]))
                                ax.plot(self.wavelength, (transmit_angle_ball[th_index,ph_index,:]+transmit_angle_diff[th_index,ph_index,:])/I, linewidth=1, color='darkred', label='T_total')
                                ax.plot(self.wavelength, transmit_angle_ball[th_index,ph_index,:]/I, linewidth=1, linestyle='dashed', color='red', label='T_ballistic')
                                ax.plot(self.wavelength, transmit_angle_diff[th_index,ph_index,:]/I, linewidth=1, linestyle='dashed', color='lightcoral', label='T_diffuse')
                                ax.set_xlabel('Wavelength (nm)')
                                ax.set_title('Angular Transmission @ theta: ' + str(int(self.theta_out_BTDF_center[th_index]*180/np.pi)) + ' phi: ' + str(int(self.phi_out_BSDF[ph_index]*180/np.pi)))
                                ax.legend()
                                plt.savefig(directory_output + '/plots/' + identifier + '_Angular_transmission_spectrum_th' + str(int(self.theta_out_BTDF_center[th_index]*180/np.pi))\
                                            + '_ph' + str(int(self.phi_out_BSDF[ph_index]*180/np.pi)))
                                plt.close()
                            else:
                                th_index = np.argmin(np.abs(self.theta_out_BRDF_center - self.angle_for_spectral_plots[n_angle,0]))
                                ax.plot(self.wavelength, (reflect_angle_spec[th_index,ph_index,:]+reflect_angle_diff[th_index,ph_index,:])/I, linewidth=1, color='darkblue', label='R_total')
                                ax.plot(self.wavelength, reflect_angle_spec[th_index,ph_index,:]/I, linewidth=1, linestyle='dashed', color='blue', label='R_specular')
                                ax.plot(self.wavelength, reflect_angle_diff[th_index,ph_index,:]/I, linewidth=1, linestyle='dashed', color='slateblue', label='R_diffuse')
                                ax.set_xlabel('Wavelength (nm)')
                                ax.set_title('Angular Reflection @ theta: ' + str(int(self.theta_out_BRDF_center[th_index]*180/np.pi)) + ' phi: ' + str(int(self.phi_out_BSDF[ph_index]*180/np.pi)))
                                ax.legend()
                                plt.savefig(directory_output + '/plots/' + identifier + '_Angular_reflection_spectrum_th' + str(int(self.theta_out_BRDF_center[th_index]*180/np.pi))\
                                            + '_ph' + str(int(self.phi_out_BSDF[ph_index]*180/np.pi)))
                                plt.close()

                req = comm.isend(cmd, dest=status.Get_source(), tag=num%1e4)
                req.wait()

                # Incidence Data Collection
                I[int(photon_state[0])] += 1
                Ns[int(photon_state[0])] += photon_state[7]
                if photon_state[5] == 0:
                    photon_state[5] = 2*np.pi
                for t in range(self.theta_in_BSDF.size-1):
                    if photon_state[4] >= self.theta_in_BSDF[t] and photon_state[4] < self.theta_in_BSDF[t+1]:
                        th_ind = t + 1
                        break
                for n_p in range(self.phi_in_BSDF.size):
                    if n_p != self.phi_in_BSDF.size - 1:
                        if photon_state[5] > self.phi_in_BSDF[n_p] and photon_state[5] <= self.phi_in_BSDF[n_p+1]:
                            ph_ind = np.array([n_p,n_p+1])
                            ph_proportions = np.array([np.abs(photon_state[5] - self.phi_in_BSDF[n_p+1]),np.abs(photon_state[5] - self.phi_in_BSDF[n_p])])\
                                             /np.abs(self.phi_in_BSDF[n_p+1] - self.phi_in_BSDF[n_p])
                            break
                    else:
                        if photon_state[5] > self.phi_in_BSDF[n_p] and photon_state[5] <= 2*np.pi:
                            ph_ind = np.array([n_p,0])
                            ph_proportions = np.array([np.abs(photon_state[5] - 2*np.pi),np.abs(photon_state[5] - self.phi_in_BSDF[n_p])])\
                                             /np.abs(2*np.pi - self.phi_in_BSDF[n_p])
                            break
                inc_angle[th_ind,ph_ind,int(photon_state[0])] += ph_proportions
                
                # Reflectance Data Collection
                if photon_state[2] == 0:
                    if photon_state[8] == 0:
                        R_spec[int(photon_state[0])] += photon_state[3]
                    else:
                        R_diff[int(photon_state[0])] += photon_state[3]
                    A_medium[int(photon_state[0])] += 1 - photon_state[3]
                    if photon_state[6] < 0:
                        photon_state[6] *= -1
                    elif photon_state[6] > np.pi:
                        photon_state[6] = 2*np.pi - photon_state[6]
                    for t in range(1, self.theta_out_BRDF_edge.size):
                        if photon_state[6] > self.theta_out_BRDF_edge[t-1] and photon_state[6] <= self.theta_out_BRDF_edge[t]:
                            th_ind = t
                            break
                    if photon_state[7] <= 0:
                        while photon_state[7] <= 0:
                            photon_state[7] += 2*np.pi
                    elif photon_state[7] > 2*np.pi:
                        while photon_state[7] > 2*np.pi:
                            photon_state[7] -= 2*np.pi
                    for n_p in range(self.phi_out_BSDF.size):
                        if n_p != self.phi_out_BSDF.size - 1:
                            if photon_state[7] > self.phi_out_BSDF[n_p] and photon_state[7] <= self.phi_out_BSDF[n_p+1]:
                                ph_ind = np.array([n_p,n_p+1])
                                ph_proportions = np.array([np.abs(photon_state[7] - self.phi_out_BSDF[n_p+1]),np.abs(photon_state[7] - self.phi_out_BSDF[n_p])])\
                                                 /np.abs(self.phi_out_BSDF[n_p+1] - self.phi_out_BSDF[n_p])
                                break
                        else:
                            if photon_state[7] > self.phi_out_BSDF[n_p] and photon_state[7] <= 2*np.pi:
                                ph_ind = np.array([n_p,0])
                                ph_proportions = np.array([np.abs(photon_state[7] - 2*np.pi),np.abs(photon_state[7] - self.phi_out_BSDF[n_p])])\
                                                 /np.abs(2*np.pi - self.phi_out_BSDF[n_p])
                                break
                    if photon_state[8] == 0:
                        reflect_angle_spec[th_ind,ph_ind,int(photon_state[0])] += photon_state[3]*ph_proportions
                        reflect_angle_spec[-1,:,int(photon_state[0])] = np.sum(reflect_angle_spec[-2,:,int(photon_state[0])])/self.phi_out_BSDF.size
                    else:
                        reflect_angle_diff[th_ind,ph_ind,int(photon_state[0])] += photon_state[3]*ph_proportions
                        reflect_angle_diff[-1,:,int(photon_state[0])] = np.sum(reflect_angle_diff[-2,:,int(photon_state[0])])/self.phi_out_BSDF.size
                
                # Transmittance Data Collection
                elif photon_state[2] == 1:
                    if photon_state[8] == 0:
                        T_ball[int(photon_state[0])] += photon_state[3]
                    else:
                        T_diff[int(photon_state[0])] += photon_state[3]
                    A_medium[int(photon_state[0])] += 1 - photon_state[3]
                    if photon_state[6] < 0:
                        photon_state[6] *= -1
                    elif photon_state[6] > np.pi:
                        photon_state[6] = 2*np.pi - photon_state[6]
                    for t in range(self.theta_out_BTDF_edge.size-1):
                        if photon_state[6] >= self.theta_out_BTDF_edge[t] and photon_state[6] < self.theta_out_BTDF_edge[t+1]:
                            th_ind = t + 1
                            break
                    if photon_state[7] <= 0:
                        while photon_state[7] <= 0:
                            photon_state[7] += 2*np.pi
                    elif photon_state[7] > 2*np.pi:
                        while photon_state[7] > 2*np.pi:
                            photon_state[7] -= 2*np.pi
                    for n_p in range(self.phi_out_BSDF.size):
                        if n_p != self.phi_out_BSDF.size - 1:
                            if photon_state[7] > self.phi_out_BSDF[n_p] and photon_state[7] <= self.phi_out_BSDF[n_p+1]:
                                ph_ind = np.array([n_p,n_p+1])
                                ph_proportions = np.array([np.abs(photon_state[7] - self.phi_out_BSDF[n_p+1]),np.abs(photon_state[7] - self.phi_out_BSDF[n_p])])\
                                                 /np.abs(self.phi_out_BSDF[n_p+1] - self.phi_out_BSDF[n_p])
                                break
                        else:
                            if photon_state[7] > self.phi_out_BSDF[n_p] and photon_state[7] <= 2*np.pi:
                                ph_ind = np.array([n_p,0])
                                ph_proportions = np.array([np.abs(photon_state[7] - 2*np.pi),np.abs(photon_state[7] - self.phi_out_BSDF[n_p])])\
                                                 /np.abs(2*np.pi - self.phi_out_BSDF[n_p])
                                break
                    if photon_state[8] == 0:
                        transmit_angle_ball[th_ind,ph_ind,int(photon_state[0])] += photon_state[3]*ph_proportions
                        transmit_angle_ball[0,:,int(photon_state[0])] = np.sum(transmit_angle_ball[1,:,int(photon_state[0])])/self.phi_out_BSDF.size
                    else:
                        transmit_angle_diff[th_ind,ph_ind,int(photon_state[0])] += photon_state[3]*ph_proportions
                        transmit_angle_diff[0,:,int(photon_state[0])] = np.sum(transmit_angle_diff[1,:,int(photon_state[0])])/self.phi_out_BSDF.size
                
                # Miscellaneous
                elif photon_state[2] == 2:
                    A_particle[int(photon_state[0])] += 1
                elif photon_state[2] == 3:
                    A_TIR[int(photon_state[0])] += 1
                elif photon_state[2] == 4:
                    R_scat[int(photon_state[0])] += photon_state[3]
                elif photon_state[2] == 5:
                    T_scat[int(photon_state[0])] += photon_state[3]
            
            cmd = 0
            for i in range(1, subgroup):
                comm.send(cmd, dest=(rank//subgroup)*subgroup+i, tag=i%1e4)
                
            np.savez(directory_output + "/data/" + identifier + "_MC_" + str(rank), I=I, R_spec=R_spec, R_diff=R_diff, R_scat=R_scat, T_ball=T_ball, T_diff=T_diff, T_scat=T_scat,
                     A_medium=A_medium, A_particle=A_particle, A_TIR=A_TIR, Ns=Ns, inc_angle=inc_angle,
                     reflect_angle_spec=reflect_angle_spec, reflect_angle_diff=reflect_angle_diff,
                     transmit_angle_ball=transmit_angle_ball, transmit_angle_diff=transmit_angle_diff)
        else:
            while True:
                photon_state = np.zeros(9).astype(np.float64)
                req = comm.irecv(source=(rank//subgroup)*subgroup, tag=MPI.ANY_TAG)
                cmd = req.wait(status)
                
                if cmd == 0:
                    break
                num = status.Get_tag()

                pht = photon(self.layer, self.wavelength, self.init_theta, self.init_phi, self.boundary[0], self.polarization)
                photon_state[0] = pht.index
                photon_state[4] = pht.theta_inc
                photon_state[5] = pht.phi_inc
    
                self.layer_change = 1
                if self.layer_crossing(pht):
                    self.interface(pht)
                while pht.exit == 0:
                    self.layer_change = 1
                    while self.layer_change:
                        self.propagation(pht)
                    if pht.exit:
                        break
                    self.particle(pht)

                photon_state[1] = pht.index
                if pht.particle_abs == 1:
                    photon_state[2] = 2
                elif pht.TIR_abs == 1:
                    photon_state[2] = 3
                elif pht.scat_abs == 1:
                    photon_state[3] = pht.weight
                    
                    if np.random.rand() < pht.pR:
                        photon_state[2] = 4
                    else:
                        photon_state[2] = 5
                else:
                    photon_state[3] = pht.weight
                    
                    if pht.z_exit > self.boundary[0]:
                        photon_state[2] = 0
                    elif pht.z_exit < 0:
                        photon_state[2] = 1
                    photon_state[6] = pht.theta_exit
                    photon_state[7] = pht.phi_exit
                photon_state[8] = pht.particle_scat
                    
                req = comm.Isend(photon_state, dest=(rank//subgroup)*subgroup, tag=num%1e4)
                req.wait()
