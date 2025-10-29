import numpy as np

def sph2cart(r,theta,phi,degrees=False):
    # si theta y phi están expresados en grados,
    # convierte a radiantes 
    if degrees:
        theta = theta*np.pi/180
        phi = phi*np.pi/180
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x,y,z

def cart2sph(x,y,z):
    eps = np.finfo(float).eps
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/(r+eps))
    phi = np.arctan2(y, x)

    return r,theta,phi