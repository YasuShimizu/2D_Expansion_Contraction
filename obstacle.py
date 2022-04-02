from calendar import WEDNESDAY
import numpy as np
import csv
from numba import jit

#@jit(nopython=True)
def ob_ini(ijh,nx,ny):
  fopen=open('obst.dat','r')
  dataReader=csv.reader(fopen)
  d1=next(dataReader); nobst=int(d1[0])
  i1=np.zeros(nobst,dtype=int);i2=np.zeros_like(i1)
  j1=np.zeros_like(i1);j2=np.zeros_like(i1)
  for n in np.arange(0,nobst):
    lp=next(dataReader)
    i1[n]=int(lp[0]);i2[n]=int(lp[1]);j1[n]=int(lp[2]);j2[n]=int(lp[3])
#    print(i1[n],i2[n],j1[n],j2[n])
    for i in np.arange(0,nx+1):
      for j in np.arange(0,ny+1):
         if i>i1[n] and i<=i2[n] and j>j1[n] and j<=j2[n]:
          ijh[i,j]=1

  return ijh

# nobst: 障害物の個数
# i1,i2,j1,j2 : 障害物のx,y方向の範囲

#@jit(nopython=True) #条件入力(if jo_tyke==2の場合)
def ob_cond(ijh,nx,ny,jo_type,jo_method,igs,jgs,num_dike,dike_dis,f_dike_pos,spos):
  if jo_method>=1:  #左右岸どちらか

    if jo_method==1 or jo_method>=3: #右岸水制
      for ii in np.arange(0,num_dike):
        sp_pos=float(ii)*dike_dis+f_dike_pos
        if sp_pos>=spos[nx]:
          break
        for i in np.arange(1,nx+1):
          if spos[i]>= sp_pos:
            i_pos=i
            break
        if i+1>=nx:
          break
        for ik in np.arange(i+1,i+igs+1):
            for j in np.arange(1,jgs+1):
              ijh[ik,j]=1

    if jo_method==2 or jo_method>=3: #左岸水制
      for ii in np.arange(0,num_dike):
        sp_pos=float(ii)*dike_dis+f_dike_pos
        if jo_method==4:
          sp_pos=sp_pos+dike_dis*.5
        if sp_pos>=spos[nx]:
          break
        for i in np.arange(1,nx+1):
          if spos[i]>= sp_pos:
            i_pos=i
            break
        if i+1>=nx:
          break
        for ik in np.arange(i+1,i+igs+1):
            for j in np.arange(ny,ny-jgs,-1):
              ijh[ik,j]=1
            
#  for i in np.arange(1,nx+1):
#    print(ijh[i,:])
#  exit()
  return ijh          
        
        

          


