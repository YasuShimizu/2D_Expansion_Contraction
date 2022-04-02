from numba.types.npytypes import NumpyNdIterType
import numpy as np
from numba import jit

@jit(nopython=True)
def dsdncal(nx,ny,ds,dn,dsi,dnj,rho_r,rho_s,rho_n,coss,sins,area,coss_node,sins_node,xgrid,ygrid,j_rep):
    for i in np.arange(0,nx+1):
        for j in np.arange(0,ny+1):
            ds[i,j]=np.sqrt((xgrid[i,j]-xgrid[i-1,j])**2+(ygrid[i,j]-ygrid[i-1,j])**2)
            coss[i,j]=(xgrid[i,j]-xgrid[i-1,j])/ds[i,j]
    for i in np.arange(1,nx):
        for j in np.arange(0,ny+1):
            coss_node[i,j]=(coss[i,j]+coss[i+1,j])*.5
    for j in np.arange(0,ny+1):
        coss_node[0,j]=coss[1,j]; coss_node[nx,j]=coss[nx,j]
                
    for j in np.arange(1,ny+1):
        for i in np.arange(0,nx+1):    
            dn[i,j]=np.sqrt((xgrid[i,j]-xgrid[i,j-1])**2+(ygrid[i,j]-ygrid[i,j-1])**2)
            sins[i,j]=(ygrid[i,j]-ygrid[i-1,j])/ds[i,j]
    
    for j in np.arange(1,ny):
        for i in np.arange(0,nx+1):
            sins_node[i,j]=(sins[i,j]+sins[i,j+1])*.5
    for i in np.arange(0,nx+1):
        sins_node[i,0]=sins[i,1]; sins_node[i,ny]=sins[i,ny]
#
#    for i in np.arange(0,nx+1):
#        wid=0.
#        for j in np.arange(1,ny+1):
#            wid=wid+dn[i,j]
#
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):
            rho_r[i,j]=(coss[i,j]*(sins[i+1,j]-sins[i,j])-sins[i,j]*(coss[i+1,j]-coss[i,j]))/ds[i,j]
    if j_rep==0:
        rho_r[0]=rho_r[1]; rho_r[nx]=rho_r[nx-1]
    else:
        rho_r[0]=rho_r[nx-1]; rho_r[nx]=rho_r[1]

    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny+1):
            rho_s[i,j]=(rho_r[i,j]+rho_r[i,j-1])*.5

    for i in np.arange(1,nx+1):
        for j in np.arange(0,ny+1):
            rho_n[i,j]=(rho_r[i,j]+rho_r[i-1,j])*.5

    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            dsi[i,j]=(ds[i,j]+ds[i,j-1]+ds[i+1,j]+ds[i+1,j-1])*.25
    
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            dnj[i,j]=(dn[i,j]+dn[i-1,j]+dn[i,j+1]+dn[i-1,j+1])*.25

    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            area[i,j]=(ds[i,j]+ds[i,j-1])*(dn[i,j]+dn[i-1,j])*.25


    return ds,dn,dsi,dnj,rho_r,rho_s,rho_n,area,coss_node,sins_node
    
@jit(nopython=True)
def center(nx,nym,dsi,spos_c):
    spos_c[1]=0.
    for i in np.arange(2,nx+1):
        spos_c[i]=spos_c[i-1]+dsi[i-1,nym]
    return spos_c

