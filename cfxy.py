import numpy as np
from numba import jit

@jit
def cfxc(cfx,nx,ny,hs,un,g,snm,v_up,hs_up,hmin):
    for i in np.arange(1,nx):
       for j in np.arange(1,ny+1):
            if hs_up[i,j]<hmin:
                cfx[i,j]=0.
            else:
                cfx[i,j]=-g*snm**2*un[i,j]*np.sqrt(un[i,j]**2+v_up[i,j]**2)/hs_up[i,j]**(4./3.)
    return cfx

@jit
def cfyc(cfy,nx,ny,hs,vn,g,snm,u_vp,hs_vp,hmin):
    for i in np.arange(1,nx):
       for j in np.arange(1,ny):
            if hs_vp[i,j]<hmin:
               cfy[i,j]=0.
            else:
                cfy[i,j]=-g*snm**2*vn[i,j]*np.sqrt(vn[i,j]**2+u_vp[i,j]**2)/hs_vp[i,j]**(4./3.)
    return cfy

@jit
def centrix(ctrx,nx,ny,un,v_up,rho_s):
    for i in np.arange(1,nx):
       for j in np.arange(1,ny+1):
           ctrx[i,j]=un[i,j]*v_up[i,j]*rho_s[i,j]
    return ctrx

@jit
def centriy(ctry,nx,ny,u_vp,rho_n):
    for i in np.arange(1,nx):
       for j in np.arange(1,ny):
           ctry[i,j]=-u_vp[i,j]**2*rho_n[i,j]
    return ctry