from numba.types.npytypes import NumpyNdIterType
import numpy as np
from numba import jit

@jit(nopython=True)
def fqbs_cal(fqbs,qbs,nx,ny,dn,hs_up,hmin,ijh,j_rep):
    for i in np.arange(0,nx):
        for j in np.arange(1,ny+1):
            if hs_up[i,j]<hmin or ijh[i,j]==1 or ijh[i+1,j]==1:
                fqbs[i,j]=0.
            else:
                fqbs[i,j]=qbs[i,j]*dn[i,j]
    if j_rep==0:
        fqbs[0]=fqbs[1]; fqbs[nx]=fqbs[nx-1]
    else:
        fqbs[0]=fqbs[nx-1]; fqbs[nx]=fqbs[1]
    return fqbs

@jit
def fqbn_cal(fqbn,qbn,nx,ny,ds,hs_vp,hmin,ijh,j_rep):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            if hs_vp[i,j]<hmin or ijh[i,j]==1 or ijh[i,j+1]==1:
                fqbn[i,j]=0.
            else:
                fqbn[i,j]=qbn[i,j]*ds[i,j]
    if j_rep==0:
        fqbn[0]=fqbn[1]; fqbn[nx]=fqbn[nx-1]
    else:
        fqbn[0]=fqbn[nx-1]; fqbn[nx]=fqbn[1]
    return fqbn

@jit
def eta_cal(eta,fqbs,fqbn,nx,ny,area,hmin,dt,dstm,ijh,hs,h,deta,eta0,j_rep):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if ijh[i,j]==0:
                div=fqbs[i,j]-fqbs[i-1,j]+fqbn[i,j]-fqbn[i,j-1]
                eta[i,j]=eta[i,j]-dstm*div*dt/area[i,j]
                hs[i,j]=h[i,j]-eta[i,j]
                if hs[i,j]<=hmin:
                    hs[i,j]=hmin
                    h[i,j]=eta[i,j]+hmin
                deta[i,j]=eta[i,j]-eta0[i,j]
    if j_rep==1:
        deta[nx]=deta[1]
        eta[nx]=eta0[nx]+deta[nx]
        
    return eta,deta,h,hs



