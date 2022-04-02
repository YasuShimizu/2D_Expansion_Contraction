import numpy as np
from numba import jit

@jit
def v_up_c(v_up,v,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            v_up[i,j]=(v[i,j]+v[i,j-1]+v[i+1,j]+v[i+1,j-1])*.25
    return v_up

@jit
def hs_up_c(hs_up,hs,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            hs_up[i,j]=(hs[i,j]+hs[i+1,j])*.5
    hs_up[0]=hs[1];hs_up[nx]=hs[nx]
    return hs_up

@jit
def u_vp_c(u_vp,u,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            u_vp[i,j]=(u[i,j]+u[i-1,j]+u[i,j+1]+u[i-1,j+1])*.25
    return u_vp

@jit
def hs_vp_c(hs_vp,hs,nx,ny):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            hs_vp[i,j]=(hs[i,j]+hs[i,j+1])*.5
    return hs_vp
