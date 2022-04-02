import numpy as np
import math,csv
from numba import jit

def center(nx,t0_degree,s0,x0,y0,lam,dds,xpos,ypos,thepos,spos):
    theta0=np.radians(t0_degree)
    s=s0; x=x0; y=y0
    xpos[0]=x; ypos[0]=y;spos[0]=s
    thepos[0]=theta0*np.sin(2.*np.pi*s/lam)
    for i in np.arange(1,nx+1):
        for j in np.arange(1,11):
            s=s+dds
            theta=theta0*np.sin(2.*np.pi*s/lam)
            x=x+dds*np.cos(theta)
            y=y+dds*np.sin(theta)
        xpos[i]=x; ypos[i]=y; thepos[i]=theta; spos[i]=s
    return xpos,ypos,thepos,spos

def czset(nx,spos,slope,eta_upe,zpos,zpos0):
    for i in np.arange(0,nx+1):
        zpos[i]=eta_upe-spos[i]*slope
        zpos0[i]=zpos[i]
    return zpos,zpos0

def czread(nx,spos,zpos,bed_file):
    fopen=open(bed_file,'r')
    dataReader=csv.reader(fopen)
    d1=next(dataReader); npos=int(d1[0])
    sloc=np.zeros(npos,dtype=float)
    zloc=np.zeros_like(sloc)
    for n in np.arange(0,npos):
        lp=next(dataReader)
        sloc[n]=float(lp[0]);zloc[n]=float(lp[1])
#        print(sloc[n],zloc[n])
    fopen.close()

    slope_up=(zloc[0]-zloc[1])/(sloc[1]-sloc[0])
    slope_dw=(zloc[npos-2]-zloc[npos-1])/(sloc[npos-1]-sloc[npos-2])
    
    for i in np.arange(0,nx+1):
        ihit=0
        for m in np.arange(0,npos-1):
            if spos[i]>=sloc[m] and spos[i]<=sloc[m+1]:
                dd=(spos[i]-sloc[m])/(sloc[m+1]-sloc[m])
                zpos[i]=zloc[m]+(zloc[m+1]-zloc[m])*dd
                ihit=1
        if ihit==0:
            if spos[i]>sloc[npos-1]:
                dd=(spos[i]-sloc[npos-1])/(sloc[npos-1]-sloc[npos-2])
                zpos[i]=zloc[npos-1]+dd*(zloc[npos-1]-zloc[npos-2])
                ihit=2
#        print(spos[i],zpos[i],ihit)
    return zpos,slope_up,slope_dw

def czread0(nx,spos,zpos,bed_file):
    fopen=open(bed_file,'r')
    dataReader=csv.reader(fopen)
    d1=next(dataReader); npos=int(d1[0])
    sloc=np.zeros(npos,dtype=float)
    zloc=np.zeros_like(sloc)
    for n in np.arange(0,npos):
        lp=next(dataReader)
        sloc[n]=float(lp[0]);zloc[n]=float(lp[1])
#        print(n,zloc[n])
    
    z_max=np.max(zloc); z_min=np.min(zloc)
    return z_max,z_min

def sggrid(nx,chb,chl,slope,eta_upe,lam,amp,delta,xpos,ypos,spos,zpos,zpos0,thepos,ny,dz,xr,yr,xl,yl,xgrid,ygrid,zgrid,zgrid0,beta_0,amp_0,j_exp,xb1,xb2,xb3,br0,br1,br2,br3,br4):
    eta0=chl*slope
    for i in np.arange(0,nx+1):
        zcenter=eta_upe-spos[i]*slope
#        print(i,zpos[i],zcenter)
        if j_exp==1:
            if spos[i]<=xb1:
                ss=spos[i]/xb1
                srb=br0+ss*(br1-br0)
                chb0=chb*srb
            elif spos[i]<=xb2:
                ss=(spos[i]-xb1)/(xb2-xb1)
                srb=br1+ss*(br2-br1)
                chb0=chb*srb
            elif spos[i]<=xb3:
                ss=(spos[i]-xb2)/(xb3-xb2)
                srb=br2+ss*(br3-br2)
                chb0=chb*srb
            else:
                ss=(spos[i]-xb3)/(chl-xb3)
                srb=br3+ss*(br4-br3)
                chb0=chb*srb
        else:
            chb0=chb
        xr[i]=xpos[i]+chb0/2.*np.sin(thepos[i])
        yr[i]=ypos[i]-chb0/2.*np.cos(thepos[i])
        xl[i]=xpos[i]-chb0/2.*np.sin(thepos[i])
        yl[i]=ypos[i]+chb0/2.*np.cos(thepos[i])
        beta=-beta_0*np.cos(2.*np.pi*(spos[i]-delta)/lam)

        for j in np.arange(0,ny+1):
            ss=float(j)/float(ny)
            xgrid[i,j]=xr[i]+ss*(xl[i]-xr[i])
            ygrid[i,j]=yr[i]+ss*(yl[i]-yr[i])
            if beta>0:
                if ss<beta:
                    dz[i,j]=amp_0
                else:
                    cosp=np.cos(np.pi*(ss-beta)/(1.-beta))
                    dz[i,j]=-amp*np.cos(2.*np.pi/lam*(spos[i]-delta))*cosp
            else:
                a_beta=np.abs(beta)
                if ss>1.-a_beta:
                    dz[i,j]=amp_0
                else:
                    cosp=np.cos(np.pi*(ss)/(1.-a_beta))
                    dz[i,j]=-amp*np.cos(2.*np.pi/lam*(spos[i]-delta))*cosp        
            zgrid[i,j]=zpos[i]+dz[i,j]
            zgrid0[i,j]=zpos0[i]
            dz[i,j]=zgrid[i,j]-zpos0[i]
#    f=open('tmp.csv', 'w',newline='')
#    writer = csv.writer(f)
#    writer.writerow([1,2,3])
#    for i in np.arange(0,nx+1):
#        zcenter=eta_upe-spos[i]*slope
#        writer.writerow([spos[i],zpos[i],zcenter,zgrid[i,1],dz[i,1]])

    return xr,yr,xl,yl,xgrid,ygrid,zgrid,zgrid0,dz

