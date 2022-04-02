import numpy as np
from numba import jit

@jit(nopython=True)
def us_cal(usta,tausta,ep,ep_x,u,v,hs,nx,ny,snm,g_sqrt,hmin,ep_alpha,u_cell,v_cell,vti,sgd):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            u_cell[i,j]=(u[i,j]+u[i-1,j])*.5
            v_cell[i,j]=(v[i,j]+v[i,j-1])*.5
            vti[i,j]=np.sqrt(u_cell[i,j]**2+v_cell[i,j]**2)
            if hs[i,j]>hmin:
                usta[i,j]=snm*g_sqrt*vti[i,j]/hs[i,j]**(1./6.)
                ep[i,j]=.4/6.*usta[i,j]*hs[i,j]*ep_alpha
                tausta[i,j]=usta[i,j]**2/sgd
            else:
                usta[i,j]=0.
                ep[i,j]=0.
                tausta[i,j]=0.
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            ep_x[i,j]=(ep[i,j]+ep[i+1,j]+ep[i,j+1]+ep[i+1,j+1])*.25
        ep_x[i,0]=(ep[i,j+1]+ep[i+1,j+1])*.5
        ep_x[i,ny]=(ep[i,j]+ep[i+1,j])*.5
    return usta,tausta,ep,ep_x,vti

@jit(nopython=True)
def qb_cal(nx,ny,tausta,qb_cell,tsc,sq3,hs,hmin):
    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            if tausta[i,j]>tsc and hs[i,j]>hmin:
                qb_cell[i,j]=8.*(tausta[i,j]-tsc)**1.5*sq3
            else:
                qb_cell[i,j]=0.
    return qb_cell            

@jit(nopython=True)
def qbs_cal(nx,ny,u,v_up,hs_up,qb_cell,vti,qbs,tausta,tsc,nsta,gamma0,eta,dsi,rho_s,rhos_s,j_rep,hmin,vmin):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny+1):
            ts_up=(tausta[i,j]+tausta[i+1,j])*.5
            qb_up=(qb_cell[i,j]+qb_cell[i+1,j])*.5
            vti_up=(vti[i,j]+vti[i+1,j])*.5
            if ts_up>tsc and qb_up>0. and abs(vti_up)>vmin and hs_up[i,j]>hmin:
                gamma=gamma0*np.sqrt(1./ts_up)
                deds=(eta[i+1,j]-eta[i,j])/dsi[i,j]
                rho=rho_s[i,j]+rhos_s[i,j]                
                qbs[i,j]=((u[i,j]-v_up[i,j]*nsta*hs_up[i,j]*rho)/vti_up-gamma*deds)*qb_up
            else:
                qbs[i,j]=0.
    if j_rep==0:
        qbs[0]=qbs[1]; qbs[nx]=qbs[nx-1]
    else:
        qbs[0]=qbs[nx-1]; qbs[nx]=qbs[1]
    return qbs

@jit(nopython=True)
def qbn_cal(nx,ny,u_vp,v,hs_vp,qb_cell,vti,qbn,tausta,tsc,nsta,gamma0,eta,dnj,rho_n,rhos_n,j_rep,hmin,vmin):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            ts_vp=(tausta[i,j]+tausta[i,j+1])*.5
            qb_vp=(qb_cell[i,j]+qb_cell[i,j+1])*.5
            vti_vp=(vti[i,j]+vti[i,j+1])*.5
            if ts_vp>tsc and qb_vp>0. and abs(vti_vp)>vmin and hs_vp[i,j]>hmin:
                gamma=gamma0*np.sqrt(1./ts_vp)
                dedn=(eta[i,j+1]-eta[i,j])/dnj[i,j]
                rho=rho_n[i,j]+rhos_n[i,j]
                qbn[i,j]=((v[i,j]+u_vp[i,j]*nsta*hs_vp[i,j]*rho)/vti_vp-gamma*dedn)*qb_vp
            else:
                qbn[i,j]=0.
    if j_rep==0:
        qbn[0]=qbn[1]; qbn[nx]=qbn[nx-1]
    else:
        qbn[0]=qbn[nx-1]; qbn[nx]=qbn[1]
    return qbn

