import numpy as np
from numba import jit

@jit
def un_cal(un,u,nx,ny,dsi,cfx,ctrx,hn,g,dt,hmin,hs,eta,ijh):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if ijh[i,j]==1 or ijh[i+1,j]==1:
                dhdx=0.; u[i,j]=0.
            elif eta[i,j]>eta[i+1,j] and hs[i,j]<=hmin and hn[i+1,j]<eta[i,j] and u[i,j]>=0.:
                dhdx=0.
            elif eta[i,j]<eta[i+1,j] and hs[i+1,j]<=hmin and hn[i,j]<eta[i+1,j]and u[i,j]<0.:
                dhdx=0.
            else:
                dhdx=(hn[i+1,j]-hn[i,j])/dsi[i,j]
            un[i,j]=u[i,j]+(ctrx[i,j]+cfx[i,j]-g*dhdx)*dt      
    return un

@jit
def vn_cal(vn,v,nx,ny,dnj,cfy,ctry,hn,g,dt,hmin,hs,eta,ijh):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            if ijh[i,j]==1 or ijh[i,j+1]==1:
                dhdy=0.; v[i,j]=0.
            elif eta[i,j]>eta[i,j+1] and hs[i,j]<=hmin and hn[i,j+1]<eta[i,j] and v[i,j]>=0.:
                dhdy=0.
            elif eta[i,j]<eta[i,j+1] and hs[i,j+1]<=hmin and hn[i,j]<eta[i,j+1] and v[i,j]<=0.:
                dhdy=0.
            else:
                dhdy=(hn[i,j+1]-hn[i,j])/dnj[i,j]
            vn[i,j]=v[i,j]+(ctry[i,j]+cfy[i,j]-g*dhdy)*dt
    return vn

@jit
def qu_cal(qu,qc,un,nx,ny,dn,hs_up,hmin,ijh):
    for i in np.arange(1,nx):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            if hs_up[i,j]<hmin or ijh[i,j]==1 or ijh[i+1,j]==1:
                qu[i,j]=0.
            else:
                qu[i,j]=un[i,j]*dn[i,j]*hs_up[i,j]
                qc[i]=qc[i]+qu[i,j]
    return qu,qc

@jit
def qv_cal(qv,vn,nx,ny,ds,hs_vp,hmin,ijh):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            if hs_vp[i,j]<hmin or ijh[i,j]==1 or ijh[i,j+1]==1:
               qv[i,j]=0.
            else:
                qv[i,j]=vn[i,j]*ds[i,j]*hs_vp[i,j]
    return qv

@jit
def qc_cal(qu,qc,nx,ny):
    for i in np.arange(1,nx):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            qc[i]=qc[i]+qu[i,j]
    qc[nx]=qc[nx-1]
    

    return qc
    