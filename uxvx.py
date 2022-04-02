import numpy as np
from numba import jit

@jit
def uv(ux,vx,uv2,hx,hsx,u,v,h,hs,nx,ny,coss_node,sins_node):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            ux[i,j]=(u[i,j]+u[i,j+1])*.5
        ux[i,0]=u[i,1]; ux[i,ny]=u[i,ny]
    
    
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            vx[i,j]=(v[i,j]+v[i+1,j])*.5
        vx[0,j]=v[1,j]; vx[nx,j]=v[nx,j]

    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            utmp=ux[i,j]*coss_node[i,j]-vx[i,j]*sins_node[i,j]
            vtmp=ux[i,j]*sins_node[i,j]+vx[i,j]*coss_node[i,j]
            ux[i,j]=utmp; vx[i,j]=vtmp

    uv2[:,:]=np.sqrt(ux[:,:]**2+vx[:,:]**2)

    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            hx[i,j]=(h[i,j]+h[i+1,j]+h[i,j+1]+h[i+1,j+1])*.25
            hsx[i,j]=(hs[i,j]+hs[i+1,j]+hs[i,j+1]+hs[i+1,j+1])*.25
        hx[i,0]=(h[i,1]+h[i+1,1])*.5
        hx[i,ny]=(h[i,ny]+h[i+1,ny])*.5
        hsx[i,0]=(hs[i,1]+hs[i+1,1])*.5
        hsx[i,ny]=(hs[i,ny]+hs[i+1,ny])*.5
    for j in np.arange(1,ny):
        hx[0,j]=(h[1,j]+h[1,j+1])*.5
        hx[nx,j]=(h[nx,j]+h[nx,j+1])*.5
        hsx[0,j]=(hs[1,j]+hs[1,j+1])*.5
        hsx[nx,j]=(hs[nx,j]+hs[nx,j+1])*.5
    hx[0,0]=h[1,1];hx[0,ny]=h[1,ny];hx[nx,0]=h[nx,1];hx[nx,ny]=h[nx,ny]
    hsx[0,0]=hs[1,1];hsx[0,ny]=hs[1,ny];hsx[nx,0]=hs[nx,1];hsx[nx,ny]=hs[nx,ny]
    return ux,vx,uv2,hx,hsx

@jit
def vortex(vor,ux,vx,nx,ny,ds,dn):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vor[i,j]=(ux[i,j+1]-ux[i,j-1])/(dn[i,j+1]+dn[i,j])- \
                     (vx[i+1,j]-vx[i-1,j])/(ds[i,j]+ds[i+1,j])

    return vor

@jit
def qbsqbn(qbx,qby,qbs_node,qbn_node,qb_node,qb_cell,qbs,qbn,nx,ny,coss_node,sins_node):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            qbs_node[i,j]=(qbs[i,j]+qbs[i,j+1])*.5
    qbs_node[:,0]=qbs[:,1]; qbs_node[:,ny]=qbs[:,ny]
        
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            qbn_node[i,j]=(qbn[i,j]+qbn[i+1,j])*.5
    qbn_node[0]=qbn[1]; qbn_node[nx]=qbn[nx]

    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            qbx[i,j]=qbs_node[i,j]*coss_node[i,j]-qbn_node[i,j]*sins_node[i,j]
            qby[i,j]=qbs_node[i,j]*sins_node[i,j]+qbn_node[i,j]*coss_node[i,j]    
    qb_node[:,:]=np.sqrt(qbx[:,:]**2+qby[:,:]**2)

    return qbx,qby,qb_node

@jit
def detacal(nx,ny,eta,deta,eta_node,deta_node):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            deta_node[i,j]=(deta[i,j]+deta[i+1,j]+deta[i,j+1]+deta[i+1,j+1])*.25
            eta_node[i,j]=(eta[i,j]+eta[i+1,j]+eta[i,j+1]+eta[i+1,j+1])*.25
        deta_node[i,0]=(deta[i,1]+deta[i+1,1])*.5
        deta_node[i,ny]=(deta[i,ny]+deta[i+1,ny])*.5
        eta_node[i,0]=(eta[i,1]+eta[i+1,1])*.5
        eta_node[i,ny]=(eta[i,ny]+eta[i+1,ny])*.5
    for j in np.arange(1,ny):
        deta_node[0,j]=(deta[1,j]+deta[1,j+1])*.5
        deta_node[nx,j]=(deta[nx,j]+deta[nx,j+1])*.5
        eta_node[0,j]=(eta[1,j]+eta[1,j+1])*.5
        eta_node[nx,j]=(eta[nx,j]+eta[nx,j+1])*.5
    deta_node[0,0]=deta[1,1];deta_node[0,ny]=deta[1,ny]
    deta_node[nx,0]=deta[nx,1];deta_node[nx,ny]=deta[nx,ny]
    eta_node[0,0]=eta[1,1];eta_node[0,ny]=eta[1,ny]
    eta_node[nx,0]=eta[nx,1];eta_node[nx,ny]=eta[nx,ny]
    return eta_node,deta_node

