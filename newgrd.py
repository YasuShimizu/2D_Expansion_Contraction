import numpy as np
from numba import jit

@jit(nopython=True)
def ng_u(gux,guy,u,un,nx,ny,dsi,dn):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            gux[i,j]=gux[i,j]+(un[i+1,j]-un[i-1,j]-u[i+1,j]+u[i-1,j])/dsi[i,j]*.5
    for i in np.arange(0,nx+1):
        for j in np.arange(2,ny):    
            guy[i,j]=guy[i,j]+(un[i,j+1]-un[i,j-1]-u[i,j+1]+u[i,j-1])/dn[i,j]*.5
    return gux,guy

@jit(nopython=True)
def ng_v(gvx,gvy,v,vn,nx,ny,ds,dnj):
    for i in np.arange(2,nx):
        for j in np.arange(1,ny):
            gvx[i,j]=gvx[i,j]+(vn[i+1,j]-vn[i-1,j]-v[i+1,j]+v[i-1,j])*ds[i,j]*.5
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            gvy[i,j]=gvy[i,j]+(vn[i,j+1]-vn[i,j-1]-v[i,j+1]+v[i,j-1])/dnj[i,j]*.5
    return gvx,gvy



