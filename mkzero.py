import numpy as np

def z0(fn,gxn,gyn,nx,ny):

    fn = np.zeros([nx+1, ny+1])
    gxn = np.zeros([nx+1, ny+1])
    gyn = np.zeros([nx+1, ny+1])
 
    return fn, gxn, gyn
