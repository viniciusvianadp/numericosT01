
# MAP5725
# based on Roma's program.

# general explicit one-Step methods and convergence tests implementation.

# (manufactured) problem with kwnown exact solution 
#            (1) I1' = -4I1 + 3I2 + 6, I1(0) = 0
#            (2) I2' = -2.4I1 + 1.6I2 + 3.6, I2(0) = 0

import math
import numpy as np

#############################################################################

def phi(t, y, f):
    # define discretization function 
    return f(t, y)     # euler 

############################################################################

def f(t, y):
    # bidimensional problem
    f0 = -4*y[0] + 3*y[1] + 6
    f1 = -2.4*y[0] + 1.6*y[1] + 3.6
    
    return np.array([f0, f1])

############################################################################

def oneStepMethod(t0, y0, T, n):
    # compute approximate solution to the initial value problem
    t_n = [t0];           # time interval: t in [t0,T]
    y_n = [np.array(y0)]  # initial condition
                         
    h   = (T - t0) / n        # integration time step

    while t_n[-1] < T:
        # advance solution in time
        y_n.append( y_n[-1] + h*phi(t_n[-1], y_n[-1],f) ) 
        t_n.append(t_n[-1] + h)     # update clock
        h = min(h, T-t_n[-1])       # select new time step
    y_n = np.array(y_n)
    
    return (T - t0) / n, y_n[-1]

############################################################################

def ye1(t):
    # exact solution 
    return -3.375*math.exp(-2*t) + 1.875*math.exp(-0.4*t) + 1.5

############################################################################

def ye2(t):
    # exact solution
    return -2.25*math.exp(-2*t) + 2.25*math.exp(-0.4*t)

############################################################################

def main():
    # obtains the numerical convergence table based on parameters such as
    # inicial conditions, final time and number of steps
    
    # input math model data
    t0=0; y0=[0, 0];  # initial condition
    T=1             # final time
    
    # input numerical method data
    m=13;  h=[0]*m;   # number of cases to run. Initialize list of time steps
    yn=[y0]*m;       # initialize list of approximations
    
    print("MANUFACTURED SOLUTION VERIFICATION TABLE");

    # case loop
    for i in range(1,m+1): # run m times same code with h progressively small
        n=16*2**(i-1);     # number of time steps in i-th case
        
        h[i-1],yn[i-1]=oneStepMethod(t0,y0,T,n);
        
        # verification via manufactured solution strategy
        # convergence table to verify the method correct implementation 
        p=q=r=0;
        
        e = max(abs(ye1(T)-yn[i-1][0]), abs(ye2(T)-yn[i-1][1]))
        if i>1:
            q = abs(max(abs(ye1(T)-yn[i-2][0]), abs(ye2(T)-yn[i-2][1]))/e)
            r = h[i-2]/h[i-1];
            
            p = math.log(q)/math.log(r);
            print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h[i-1],e,p))
        else: 
            print("%5d & %9.3e & %9.3e & --------- \\\\" % (n,h[i-1],e))    
    print(" "); 
    
############################################################################
  
main()

