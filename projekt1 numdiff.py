from numpy import *
import numpy as np
from scipy import linalg
from scipy.linalg import expm
from matplotlib.pyplot import *

def RK4step(f,told,uold,h):

    Y1= f(told,uold)
    Y2 = f(told + 0.5*h,uold + h*0.5*Y1)
    Y3 = f(told + 0.5*h,uold + h*0.5*Y2)
    Y4 = f(told + h,uold + h*Y3)
    
    unew= uold + h*(Y1 + 2*Y2 + 2*Y3 + Y4)/6
    
    return unew

def RK4Int(f,y0,t0,tf,N):
    uold = y0
    told=t0
    h=(tf-t0)/N
    approx=[]
    while told<tf:
        approx.append(uold)
        utemp=RK4step(f,told,uold,h)
        told=told+h
        uold=utemp
    approx.append(uold) # Här tar lägger vi till det sista steget.
    return approx

def RK_errVSh(A,f,y0,t0,tf):
    error=[]
    h=[]
    
    truesol = expm(A*tf)@y0
    for i in range(10):
        N= 2**(i+5)
        approx=RK4Int(f,y0,t0,tf,N)
        err = linalg.norm(approx[-1]-truesol)
        error.append(err)
        h.append((tf-t0)/N)   
    loglog(h,error)
    grid(True,'both')

lambd=-1
def f(t,y):
    return -y

RK_errVSh(lambd,f,np.array([1]),0,1)


def RK34step(f,told,uold,h):
        
    Y1= f(told,uold)
    Y2 = f(told+0.5*h,uold+h*Y1*0.5)
    Y3 = f(told+0.5*h,uold+h*Y2*0.5)
    Y4 = f(told+h,uold+h*Y3)
    
    Z3 = f(told+h,uold-h*Y1+2*h*Y2)
        
    unew= uold + h*(Y1+2*Y2+2*Y3+Y4)/6
    
    err = linalg.norm((h/6)*(2*Y2+Z3-2*Y3-Y4)) #l_n+1
    return unew,err

def newstep(tol,err,errold,hold,k):
    hnew=(tol/err)**(2/3/k) * (tol/errold)**(-1/3/k) * hold
    return hnew

def adaptiveRK34(f,t0,tf,y0,tol=1e-8): #Ny version
    k=4
    hold=abs(tf-t0)*tol**(1/4)/(100*(1+linalg.norm(f(t0,y0))))
    h=hold
    told=t0
    uold=y0
    errold=tol #r0
    t=[]
    y=[]
    while (told+h)<tf:
        unew,err=RK34step(f,told,uold,h)
        uold = unew
        t.append(told)
        y.append(uold)
        told=told+h
        h=newstep(tol,err,errold,hold,k)
        errold=err
    hlast=tf-told
    ulast,err=RK34step(f,told,uold,hlast)
    y.append(ulast)
    t.append(tf)
    
    return np.array(t),np.stack(y) #Stack fixar outputen så att det blir en enda array för Lotka. 

#2.1
def LotkaVolterra(t,y):
    (a,b,c,d)=[3,9,15,15]
    dydt = [a*y[0] - b*y[0]*y[1], c*y[0]*y[1]-d*y[1]]
    return np.array(dydt)

y0=[1,1]
t0=0
tf=10

t, y = adaptiveRK34(LotkaVolterra,t0,tf,y0)
t, y = adaptiveRK34(f,t0,tf,y0)
figure()
plot(y[:,0],y[:,1]) #Plottar x mot y
figure()
plot(t,y[:,0]) # Plottar x mot t
figure()
plot(t,y[:,1]) # Plottar y mot t
