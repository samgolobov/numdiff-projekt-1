from numpy import *
from scipy import linalg
from scipy.linalg import expm
from matplotlib.pyplot import *

def RK4step(f,told,uold,h):
    B=array([[1/6],[1/3],[1/3],[1/6]])
    
    yprim1= f(told,uold)
    yprim2 = f(told+0.5*h,uold+h*yprim1*0.5)
    yprim3 = f(told+0.5*h,uold+h*yprim2*0.5)
    yprim4 = f(told+h,uold+h*yprim3)
    
    Yprim=array([yprim1,yprim2,yprim3,yprim4])
    
    unew= uold + h*((1/6)*yprim1+(1/3)*yprim2+(1/3)*yprim3+(1/6)*yprim4)
    return unew

lambd = array([[-1]])
def f(t,y):
    return lambd*y

def RK4Int(f,y0,t0,tf,N):
    uold = y0
    told=t0
    h= (tf-t0)/N
    ystore=[]
    #err=[]
    for i in range(int(N)):
        ystore.append(uold)
        utemp=RK4step(f,told,uold,h)
        told=told+h
        uold=utemp
        
    tgrid=linspace(t0,tf,N)
    approx = ystore
    return approx

def RK_errVSh(A,f,y0,t0,tf):
    error=[]
    h=[]
    for i in range(7):
        N= 2**(3+i)
        print(N)
        approx=RK4Int(f,y0,t0,tf,N)
        err = linalg.norm(approx[:][-1]-expm(A*tf)@y0)
        error.append(err)
        h.append((tf-t0)/N)   
    loglog(h,error)
    grid(True, which="both", ls="-")


RK_errVSh(lambd,f,array([1]),0,0.1)


def RK34step(f,told,uold,h):
    
    B=array([[1/6],[1/3],[1/3],[1/6]])
    
    yprim1= f(told,uold)
    yprim2 = f(told+0.5*h,uold+h*yprim1*0.5)
    yprim3 = f(told+0.5*h,uold+h*yprim1*0.5)
    yprim4 = f(told+h,uold+h*yprim1)
    
    zprim3 = f(told+h,uold-h*yprim1+2*h*yprim2)
    
    Yprim=array([yprim1,yprim2,yprim3,yprim4])
    
    unew= uold + Yprim*B
    
    err = (h/6)*(2*yprim2-zprim3+2*yprim3-yprim4) #l_n+1
    

    return unew,err

def newstep(tol,err,errold,hold,k):
    hnew=(tol/err)**(2/3/k) * (tol/errold)**(-1/3/k) * hold
    return hnew

def adaptiveRK34(f,t0,tf,y0,tol):
    tol=1e-6
    h0=abs(tf-t0)*tol**(1/4)/(100*(1+linalg.norm(f(y0))))
    h=h0
    told=t0
    uold=y0
    errold=tol #r0
    
    t=[]
    y=[]
    while (told+h)<tf:
        errold=err
        (unew,err)=RK34step(f,told,uold,h)
        uold = unew
        t.append(told)
        y.append(uold)
        h=newstep(tol,err,errold,hold,k)
        told=told+h
    hlast=tf-told
    ulast=RK34step(f,told,uold,hlast)
    y.append(ulast)
    t.append(tf)
    
    return t,y