import numpy as np
from numba import jit
@jit
def diff_u(un,uvis,uvis_x,uvis_y,nx,ny,ds,dsi,dn,dt,ep,ep_x,cw,uvis_c,rho_s):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            uvis_x[i,j]=ep[i,j]*(un[i,j]-un[i-1,j])/(ds[i,j]+ds[i,j-1])*2.
    
    for i in np.arange(0,nx):
        for j in np.arange(1,ny):
            uvis_y[i,j]=ep_x[i,j]*(un[i,j+1]-un[i,j])/(dn[i,j]+dn[i,j-1])*2.
        uvis_y[i,ny]=-cw*un[i,ny]*abs(un[i,ny])
        uvis_y[i,0]=cw*un[i,1]*abs(un[i,1])

    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            if j==1:
                uvis_c[i,j]=ep_x[i,j]*rho_s[i,j]*(un[i,j+1]-un[i,j])/dn[i,j]
            elif j==ny:
                uvis_c[i,j]=ep_x[i,j-1]*rho_s[i,j]*(un[i,j]-un[i,j-1])/dn[i,j]
            else:
                uvis_c[i,j]=(ep_x[i,j]+ep_x[i,j-1])*.5*rho_s[i,j]* \
                    (un[i,j+1]-un[i,j-1])/dn[i,j]*.5

    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            uvis[i,j]=(uvis_x[i+1,j]-uvis_x[i,j])/dsi[i,j] \
                +(uvis_y[i,j]-uvis_y[i,j-1])/dn[i,j]+uvis_c[i,j]

    
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            un[i,j]=un[i,j]+uvis[i,j]*dt
    
    return un
@jit
def diff_v(vn,vvis,vvis_x,vvis_y,nx,ny,ds,dn,dnj,dt,ep,ep_x,vvis_c,rho_n):
    for i in np.arange(1,nx):
        for j in np.arange(0,ny+1):
            vvis_x[i,j]=ep_x[i,j]*(vn[i+1,j]-vn[i,j])/(ds[i,j]+ds[i-1,j])*2.
    vvis_x[0]=vvis_x[1]; vvis_x[nx]=vvis_x[nx-1]

    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            vvis_y[i,j]=ep[i,j]*(vn[i,j]-vn[i,j-1])/(dn[i,j]+dn[i-1,j])*2.

    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny):
            vvis_c[i,j]=(ep[i,j+1]+ep_x[i,j])*.5*rho_n[i,j]* \
                (vn[i,j+1]-vn[i,j-1])/dnj[i,j]*.5

    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vvis[i,j]=(vvis_x[i,j]-vvis_x[i-1,j])/ds[i,j] \
                +(vvis_y[i,j+1]-vvis_y[i,j])/dnj[i,j]+vvis_c[i,j]
    vvis[0]=vvis[1]; vvis[nx]=vvis[nx-1]

    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            vn[i,j]=vn[i,j]+vvis[i,j]*dt
    
    return vn





