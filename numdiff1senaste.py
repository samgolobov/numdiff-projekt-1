from numpy import *
import time
import timeit
import numpy as np
from scipy import linalg
from scipy.linalg import inv
from scipy.linalg import expm
from matplotlib.pyplot import *
from scipy import integrate
from scipy.integrate import solve_ivp

#Task 1.1
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

def f1(t,y):
    return -y
RK_errVSh(np.array([[-1]]),f1,array([1]),0,1)

#Task 1.2

def RK34step(f,told,uold,h):
        
    Y1= f(told,uold)
    Y2 = f(told+0.5*h,uold+h*Y1*0.5)
    Y3 = f(told+0.5*h,uold+h*Y2*0.5)
    Y4 = f(told+h,uold+h*Y3)
    
    Z3 = f(told+h,uold-h*Y1+2*h*Y2)
        
    unew= uold + h*(Y1+2*Y2+2*Y3+Y4)/6
    
    err = linalg.norm(h*(2*Y2+Z3-2*Y3-Y4)/6) #l_n+1
    return unew,err

#Task 1.3
def newstep(tol,err,errold,hold,k):
    hnew=(tol/err)**(2/(3*k)) * (tol/errold)**(-1/(3*k)) * hold
    return hnew

#Task 1.4
def adaptiveRK34(f,t0,tf,y0,tol=1e-6):
    k=4
    hold=(abs(tf-t0)*tol**(1/4))/(100*(1+linalg.norm(f(t0,y0))))
    h=hold
    told=t0
    uold=y0
    errold=tol #r0
    t=[]
    y=[]
    N=0
    while (told+h)<tf:
        unew,err=RK34step(f,told,uold,h)
        uold = unew
        t.append(told)
        y.append(uold)
        told=told+h
        h=newstep(tol,err,errold,h,k)
        errold=err
        N+=1
    hlast=tf-told
    ulast,err=RK34step(f,told,uold,hlast)
    y.append(ulast)
    t.append(tf)
    N+=1
    return np.array(t),np.stack(y), N

#Task 2.1

def LotkaVolterra(t,y):
    (a,b,c,d)=[3,9,15,15]
    dydt = [a*y[0] - b*y[0]*y[1], c*y[0]*y[1]-d*y[1]]
    return np.array(dydt)

#f = lambda t,y: LotkaVolterra(t,y)

#y0=np.array([1,1])
y0=[1,0.33]
t0=0
tf=5

t, y, n = adaptiveRK34(f,t0,tf,y0)
figure()
plot(y[:,0],y[:,1]) #Plottar x mot y
y0=[0.9,1.3]
#t, y, n = adaptiveRK34(f,t0,tf,y0)
#plot(t,y[:,0],'.')
#plot(t,y[:,1],'.')
#y0=[0.9,2.3]
#t, y, n = adaptiveRK34(f,t0,tf,y0)
#plot(y[:,0],y[:,1],'.')



y0=[1,1]
t0=0
tf=1000
t, y, n = adaptiveRK34(f,t0,tf,y0)
def H(x,y):
    (a,b,c,d)=[3,9,15,15]
    return c*x+b*y-d*log(x)-a*log(y)
Hinv = abs(H(y[:,0],y[:,1])/H(y0[0],y0[1])-1)
loglog(t,Hinv)

def vanderPol(t,y,my):
    y1=y[0]
    y2=y[1]
    dydt=[y2, (my*(1-(y1**2))*y2)-y1]
    return np.array(dydt)

y0=[2,0]
t0=0
N=[]
my=100
f = lambda t,y: vanderPol(t,y,my=my)
t, y, n = adaptiveRK34(f,t0,tf,y0,1e-8)

E6 = np.array([10,15,22,33,47,68,100,150,220,330,470,680,1000])
y0=[2,0]
t0=0
N=[]

for i in E6:
    my = i
    tf=0.7*my
    print(i)
    f = lambda t,y: vanderPol(t,y,my=my)
    t, y, n = adaptiveRK34(f,t0,tf,y0,1e-8)
    N.append(n)
    

loglog(E6,N)
loglog(E6,N,'.')
grid(True,'both')
xlabel(r'$\mu$')
ylabel('N')

#Försök att plott linjen från polyfit
np.polyfit(log(E6[:]),log(N[:]),1)
def g(x):
    return 1.62780526*x+1.45606928
loglog(E6,N)
loglog(E6,N,'.')
x=linspace(10,1000,990)
plot(x,g(x))

y0=[2,0]
t, y, n = adaptiveRK34(f,t0,tf,y0,1e-8)
plot(t,y[:,1])
y0=[2.1,0]
t, y, n = adaptiveRK34(f,t0,tf,y0,1e-8)
plot(t,y[:,1])
y0=[1.9,0]
t, y, n = adaptiveRK34(f,t0,tf,y0,1e-8)
plot(t,y[:,1])

E6 = np.array([10,15,22,33,47,68,100,150,220,330,470,680,1000,10000])
y0=[2,0]
t0=0
N_steps=[]
for i in E6:
    my = i
    tf=0.7*my
    tspan=np.array([t0,tf])
    f = lambda t,y: vanderPol(t,y,my=my)
    OUT = solve_ivp(f,tspan,y0,method='BDF')
    N_steps.append(len(OUT.t))

loglog(E6,N_steps)
loglog(E6,N_steps,'.')
grid(True,'both')
xlabel(r'$\mu$')
ylabel('N')


my=10
tf=2*my
tspan=[t0,tf]
f = lambda t,y: vanderPol(t,y,my=my)
OUT = solve_ivp(f,tspan,y0,method='BDF')
plot(OUT.y[0,:],OUT.y[1,:],'.')
len(OUT.t)

#%matplotlib qt commando för att få interactive plot

plot(OUT.y[0,:],OUT.y[1,:]) #plottar y1 mot y2.
