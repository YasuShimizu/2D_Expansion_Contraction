from types import SimpleNamespace
from xml.dom.expatbuilder import theDOMImplementation
import numpy as np
from numba import jit

@jit
def down(nx,ny,dn,eta,qp,snm,slope,hs0):
    h_max=np.max(eta)+hs0*2.; h_min=np.min(eta)
    eps=qp;epsmin=qp/100.
#    print(eps,epsmin)
    while eps>epsmin:
        h0_dw=(h_max+h_min)*.5
        qcd=0.
        for j in np.arange(1,ny+1):
            hs1=h0_dw-eta[nx,j]
            if hs1<0.:
                hs1=0.
                u01=0.
            else:
                u01=1./snm*hs1**(2./3.)*np.sqrt(slope)
                qcd=qcd+u01*hs1*dn[nx,j]
        eps=np.abs(qp-qcd)
        if qcd>qp:
            h_max=h0_dw
        else:
            h_min=h0_dw

    width_dw=0.
    eta_dw_ave=0.
    for j in np.arange(1,ny+1):
        width_dw=width_dw+dn[nx,j]
        eta_dw_ave=eta_dw_ave+eta[nx,j]*dn[nx,j]
    eta_dw_ave=eta_dw_ave/width_dw
    hs0_dw=h0_dw-eta_dw_ave            
    return h0_dw,width_dw,eta_dw_ave,hs0_dw

@jit
def up(nx,ny,dn,eta,qp,snm,slope,hs0):
    h_max=np.max(eta)+hs0*2.; h_min=np.min(eta)
    eps=qp;epsmin=qp/100.
#    print(eps,epsmin)
    while eps>epsmin:
        h0_up=(h_max+h_min)*.5
        qcd=0.
        for j in np.arange(1,ny+1):
            hs1=h0_up-eta[1,j]
            if hs1<0.:
                hs1=0.
                u01=0.
            else:
                u01=1./snm*hs1**(2./3.)*np.sqrt(slope)
                qcd=qcd+u01*hs1*dn[1,j]
        eps=np.abs(qp-qcd)
        if qcd>qp:
            h_max=h0_up
        else:
            h_min=h0_up
    
    width_up=0.
    eta_up_ave=0.
    for j in np.arange(1,ny+1):
        width_up=width_up+dn[1,j]
        eta_up_ave=eta_up_ave+eta[1,j]*dn[1,j]
    eta_up_ave=eta_up_ave/width_up
    hs0_up=h0_up-eta_up_ave            
    return h0_up,width_up,eta_up_ave,hs0_up      

@jit
def h_line(hpos_c,spos_c,h0_dw,h0_up,nx,nym,hmin,eta):
    tlen=spos_c[nx]
    for i in np.arange(1,nx+1):
        ss=spos_c[i]/tlen
        hpos_c[i]=h0_up+(h0_dw-h0_up)*ss
        hs_c=hpos_c[i]-eta[i,nym]
        if hs_c< hmin:
            hpos_c[i]=eta[i,nym]
    return hpos_c


@jit
def h_uniform(hpos_c,c_area,vel_ave,nx,ny,dn,eta,qp,snm,hs0_dw,h0_dw,hmin,slope,g):
    hpos_c[nx]=h0_dw
    epsmin=qp/1000.
    for i in np.arange(1,nx):
        h_max=np.max(eta[i,:])+hs0_dw*2.; h_min=np.min(eta[i,:])
        eps=qp
        w_width=0.
        while eps>epsmin:
            hpos_c[i]=(h_max+h_min)*.5
            qcd=0.
            c_area[i]=0.;w_width=0.
            for j in np.arange(1,ny+1):
                hs1=hpos_c[i]-eta[i,j]
                if hs1<hmin:
                    hs1=0.
                    u01=0.
                else:
                    u01=1./snm*hs1**(2./3.)*np.sqrt(slope)
                    dni=(dn[i,j]+dn[i-1,j])*.5
                    w_width=w_width+dni
                    c_area[i]=c_area[i]+dni*hs1
                    qcd=qcd+u01*hs1*dni
            eps=np.abs(qp-qcd)
            if qcd>qp:
                h_max=hpos_c[i]
            else:
                h_min=hpos_c[i]
        ave_dep=c_area[i]/w_width
        vel_ave[i]=qp/c_area[i]
        fr_num=vel_ave[i]/np.sqrt(g*ave_dep)
#        print(i,vel_ave[i],ave_dep,fr_num,w_width)
    return hpos_c


#@jit
def h_nonuni(hpos_c,c_area,vel_ave,e_slope,alf_f,eta,qp,spos_c,hs0_dw,h0_dw,nx,ny,nym,ds,dn,snm,hmin,g):
    hpos_c[nx]=h0_dw
    epsmin=hmin
    for i in np.arange(nx,1,-1):
        c_area[i]=0.;b1=0.; b2=0.; w_width=0.
        for j in np.arange(1,ny+1):
            hs1=hpos_c[i]-eta[i,j]
            if hs1>hmin:
                dnn=(dn[i,j]+dn[i-1,j])*.5
                w_width=w_width+dnn
                c_area[i]=c_area[i]+hs1*dnn
                b1=b1+dnn*hs1**3/snm**3
                b2=b2+dnn*hs1**(5./3.)/snm
        alf_f[i]=b1/b2**3
        e_slope[i]=qp**2/b2**2
        vel_ave[i]=qp/c_area[i]
        ave_dep=c_area[i]/w_width
        fr_num=vel_ave[i]/np.sqrt(g*ave_dep)
#        print(i,ave_dep,vel_ave[i],fr_num)
        if fr_num >1.0:
            bed_ave=np.mean(eta[i,:])
#            print(eta[i,1],bed_ave)

        if i>1:
            dsx=(ds[i,nym]+ds[i-1,nym])*.5
            sslope=(eta[i-1,nym]-eta[i,nym])/dsx
            hpos_c[i-1]=hpos_c[i]+dsx*sslope
            eps=hs0_dw; nc=0
            while eps>epsmin and nc<500:
                c_area[i-1]=0.
                b1=0.;b2=0.
                for j in np.arange(1,ny+1):
                    hs1=hpos_c[i-1]-eta[i-1,j]
                    if hs1>hmin:
                        dnn=(dn[i-1,j]+dn[i-2,j])*.5
                        c_area[i-1]=c_area[i-1]+hs1*dnn
                        b1=b1+dnn*hs1**3/snm**3
                        b2=b2+dnn*hs1**(5./3.)/snm
                alf_f[i-1]=b1/b2**3
                e_slope[i-1]=qp**2/b2**2
                h_a1=hpos_c[i]+qp**2/(2.*g)*(alf_f[i]-alf_f[i-1])+dsx*.5*(e_slope[i]+e_slope[i-1])
                eps=np.abs(h_a1-hpos_c[i-1])
                nc=nc+1
                hpos_c[i-1]=h_a1
    return hpos_c
