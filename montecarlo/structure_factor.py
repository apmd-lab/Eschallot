import numpy as np
from scipy.interpolate import interpn

def structure_factor(wavelength, theta_inc, phi_inc, theta_out, phi_out, f_vol, n_bg, outer_radius, particle_distribution):
    """ takes into account self interference due to high particle density
        need to input Bruggeman effective medium RI when the mie object is called
    """
    theta_out[0] = 10**-3
    wvl = wavelength.size
    th = theta_out.size
    ph = phi_out.size
    f_vol = np.sum(f_vol)
    psi = 3*f_vol/(1-f_vol)
    
    s = np.zeros((th, wvl))
    for t in range(0, th):
        q = ((4*np.pi*np.real(n_bg))/wavelength)*np.sin(theta_out[t]/2)
        qd = q[:,np.newaxis]*outer_radius[np.newaxis,:]
        f1 = np.dot(qd**3, particle_distribution)
        f2 = np.dot((np.cos(qd)+qd*np.sin(qd))*(np.sin(qd)-qd*np.cos(qd)), particle_distribution)
        f3 = np.dot((np.sin(qd)-qd*np.cos(qd))**2, particle_distribution)
        f4 = np.dot(qd**2*np.sin(qd)*np.cos(qd), particle_distribution)
        f5 = np.dot(qd**2*(np.sin(qd))**2, particle_distribution)
        f6 = np.dot(qd*np.sin(qd)*(np.sin(qd)-qd*np.cos(qd)), particle_distribution)
        f7 = np.dot(qd*np.cos(qd)*(np.sin(qd)-qd*np.cos(qd)), particle_distribution)
        
        b = psi*f2/f1
        c = psi*f3/f1
        d = 1+psi*f4/f1
        e = psi*f5/f1
        f = psi*f6/f1
        g = -psi*f7/f1
        
        X = 1+b+(2*e*f*g+d*(f**2-g**2))/(d**2+e**2)
        Y = c+(2*d*f*g-e*(f**2-g**2))/(d**2+e**2)        
        s[t,:] = ((Y/c)/(X**2+Y**2)).T
         
    theta_out[0] = 0
    
    s = s.T
    s_expand = np.zeros((wvl, th, ph))
    s_expand[:,:,:] = s[:,:,np.newaxis]
    s_interp = np.zeros((wvl, theta_inc.size, phi_inc.size, th, ph))
    for th_inc in range(theta_inc.size):
        for ph_inc in range(phi_inc.size):
            angle_out_abs = get_absolute_angles(theta_inc, phi_inc, theta_out, phi_out, th_inc, ph_inc)
            for w in range(wvl):
                arg_inc = np.concatenate((wavelength[w]*np.ones((angle_out_abs.shape[0], 1)), angle_out_abs), axis=1)
                s_temp = interpn((wavelength, theta_out, np.append(phi_out, 2*np.pi)),
                                 np.concatenate((s_expand, s_expand[:,:,0].reshape(wvl,th,1)), axis=2), arg_inc)
                s_interp[w,th_inc,ph_inc,:,:] = s_temp.reshape(th,ph)
    
    return s_interp

def get_absolute_angles(theta_inc, phi_inc, theta_out, phi_out, th_inc, ph_inc):
    th = theta_out.size
    ph = phi_out.size
    
    angle_out_abs = np.zeros((th*ph, 2)).astype(np.float64)
    for th_out in range(th):
        for ph_out in range(ph):
            v_rel = np.array([np.sin(theta_out[th_out])*np.cos(phi_out[ph_out]),
                              np.sin(theta_out[th_out])*np.sin(phi_out[ph_out]),
                              -np.cos(theta_out[th_out])])
            Rot_y = np.array([[np.cos(-theta_inc[th_inc]),0,np.sin(-theta_inc[th_inc])],
                              [0,1,0],
                              [-np.sin(-theta_inc[th_inc]),0,np.cos(-theta_inc[th_inc])]])
            Rot_z = np.array([[np.cos(phi_inc[ph_inc]),-np.sin(phi_inc[ph_inc]),0],
                              [np.sin(phi_inc[ph_inc]),np.cos(phi_inc[ph_inc]),0],
                              [0,0,1]])
            v_abs = Rot_z @ Rot_y @ v_rel
            theta_out_abs = np.arccos(-v_abs[2])
            if theta_out_abs == 0:
                phi_out_abs = 0
            else:
                phi_temp = v_abs[0]/np.sin(theta_out_abs)
                if phi_temp < -1:
                    phi_temp = -1
                elif phi_temp > 1:
                    phi_temp = 1
                if v_abs[1] >= 0:
                    phi_out_abs = np.arccos(phi_temp)
                else:
                    phi_out_abs = 2*np.pi - np.arccos(phi_temp)
            angle_out_abs[th_out*ph+ph_out,:] = np.array([theta_out_abs, phi_out_abs])
    
    return angle_out_abs
