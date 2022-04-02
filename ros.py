from numba.types.npytypes import NumpyNdIterType
import numpy as np
from numba import jit

@jit(nopython=True)
def uv_node(u_node,v_node,u,v,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            u_node[i,j]=(u[i,j]+u[i,j+1])*.5
        u_node[i,0]=u[i,1]; u_node[i,ny]=u[i,ny]
    
    
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            v_node[i,j]=(v[i,j]+v[i+1,j])*.5
        v_node[0,j]=v[1,j]; v_node[nx,j]=v[nx,j]
    return u_node,v_node

@jit(nopython=True)
def roscal(nx,ny,ds,dn,rhos_r,rhos_s,rhos_n,u,v,u_node,v_node,j_rep,rho_max,vmin):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            u2=u_node[i,j]**2; v2=v_node[i,j]**2; uv=u_node[i,j]*v_node[i,j]
            u2v2=u2+v2
            if u2v2>vmin:
                v3=(u2v2)**(3./2.)
                duds=(u_node[i+1,j]-u_node[i-1,j])/(ds[i+1,j]+ds[i,j])
#                dudn=(u_node[i,j+1]-u_node[i,j-1])/(dn[i,j+1]+dn[i,j])
#                dvds=(v_node[i+1,j]-v_node[i-1,j])/(ds[i+1,j]+ds[i,j])
                dudn=(u[i,j+1]-u[i,j])/(dn[i,j+1]+dn[i,j])*2.
                dvds=(v[i+1,j]-v[i,j])/(ds[i+1,j]+ds[i,j])*2.
                dvdn=(v_node[i,j+1]-v_node[i,j-1])/(dn[i,j+1]+dn[i,j])
                rhos_r[i,j]=(u2*dvds-v2*dudn+uv*(dvdn-duds))/v3
                rhos_r[i,j]=min(rhos_r[i,j],rho_max)
                rhos_r[i,j]=max(rhos_r[i,j],-rho_max)
            else:
                rhos_r[i,j]=0.
    if j_rep==0:
        rhos_r[0]=rhos_r[1]; rhos_r[nx]=rhos_r[nx-1]
    else:
        rhos_r[0]=rhos_r[nx-1]; rhos_r[nx]=rhos_r[1]

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            rhos_s[i,j]=(rhos_r[i,j]+rhos_r[i,j-1])*.5

    for i in np.arange(1,nx+1):
        for j in np.arange(0,ny+1):
            rhos_n[i,j]=(rhos_r[i,j]+rhos_r[i-1,j])*.5

    return rhos_r,rhos_s,rhos_n
