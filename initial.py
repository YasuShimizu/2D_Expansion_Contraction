import numpy as np
from numba import jit 

def usc(diam):
    dcm=diam*100.
    if dcm>=0.303:
        uc2=80.9*dcm
    elif dcm>=0.118:
        uc2=134.6*dcm**(31./22.)
    elif dcm>=0.0565:
        uc2=55.*dcm
    elif dcm>=0.0065:
        uc2=8.41*dcm**(11./22.)
    else:
        uc2=226.*dcm
    usta_c2=uc2/10000.
    return usta_c2




@jit(nopython=True) 
def u_init(u,qu,qc,dn,hs_up,nx,ny,snm,slope,hmin,qp):
    for i in np.arange(0,nx+1):
        qc[i]=0.
        for j in np.arange(1,ny+1):
            hss=hs_up[i,j]
            if hss>hmin:
                u[i,j]=1./snm*hss**(2./3.)*np.sqrt(slope)
            else:
                u[i,j]=0.
            qu[i,j]=u[i,j]*hss*dn[i,j]
            qc[i]=qc[i]+qu[i,j]

        qdiff=qc[i]/qp
        qc[i]=0.
        for j in np.arange(1,ny+1):
            u[i,j]=u[i,j]/qdiff
            qu[i,j]=qu[i,j]/qdiff
            qc[i]=qc[i]+qu[i,j]
    return u        

@jit(nopython=True)     
def h_init(h,hs,hpos_c,eta,nx,ny,hmin):
 for i in np.arange(1,nx+1):
     for j in np.arange(1,ny+1):
        h[i,j]=hpos_c[i]
        hs[i,j]=h[i,j]-eta[i,j]
        if hs[i,j]< hmin:
            hs[i,j]=hmin
            h[i,j]=eta[i,j]+hmin
 return h,hs

@jit(nopython=True) 
def eta_init(eta,eta0,nx,ny,ds,dn,zgrid,zgrid0):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            eta[i,j]=(zgrid[i,j]+zgrid[i-1,j]+zgrid[i,j-1]+zgrid[i-1,j-1])*.25
            eta0[i,j]=(zgrid0[i,j]+zgrid0[i-1,j]+zgrid0[i,j-1]+zgrid0[i-1,j-1])*.25
    return eta,eta0

@jit(nopython=True) 
def ep_init(ep,ep_x,nx,ny,snu_0):
    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            ep[i,j]=snu_0; ep_x[i,j]=snu_0
    return ep,ep_x

@jit(nopython=True)        
def diffs_init(gux,guy,gvx,gvy,u,v,nx,ny,dsi,dnj,ds,dn):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            gux[i,j]=(u[i+1,j]-u[i-1,j])/(2*dsi[i,j])
            gvx[i,j]=(v[i+1,j]-v[i-1,j])/(2*ds[i,j])
    
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            guy[i,j]=(u[i,j+1]-u[i,j-1])/(2*dn[i,j])
            gvy[i,j]=(v[i,j+1]-v[i,j-1])/(2*dnj[i,j])

    return gux,guy,gvx,gvy




