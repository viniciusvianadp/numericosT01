
# MAP5725
# based on Roma's program.

# general explicit one-Step methods and convergence tests implementation.

# (manufactured) problem with kwnown exact solution 
#              y"-2y'+2y = (e^t)*sin(t), 0<=t<=1, y(0) = -0.4, y'(0) = -0.6
                         
import math
import numpy as np

#############################################################################

def phi(t, y, f):
    # define discretization function 
    return f(t, y)     # euler 

############################################################################

def f(t, y):
    # bidimensional problem
    f0 =  y[1]
    f1 =  math.exp(2*t)*math.sin(t) - 2*y[0] + 2*y[1]
    
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

def ye(t):
    # exact solution 
    return 0.2*math.exp(2*t)*(math.sin(t) - 2*math.cos(t))

############################################################################
###########################################################################
def main():
    # obtains the numerical convergence table based on parameters such as
    # inicial conditions, final time and number of steps

    # input numerical model data
    t0=0; y0=[-0.4, -0.6];  # initial condition
    T=1             # final time
    
    # input numerical method data
    m=13;  h=[0]*m;   # number of cases to run. Initialize list of time steps
    yn=[y0]*m;       # initialize list of approximations
    print("MANUFACTURED SOLUTION VERIFICATION TABLE");

    # case loop
    for i in range(1,m+1): # run m times same code with h progressively small
        n=16*2**(i-1);     # number of time steps in i-th case

        h[i-1],yn[i-1]=oneStepMethod(t0,y0,T,n);

        # verification via manufactured solution stragegy
        # convergence table to verify the method correct implementation 
        e=p=q=r=0;
        if i>2:
            q = abs((ye(T)-yn[i-2][0])/(ye(T)-yn[i-1][0]));
            r = h[i-2]/h[i-1];

            p = math.log(q)/math.log(r);

            e = abs(ye(T)-yn[i-1][0])
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h[i-1],e,p));

    print(" ");



     # verification of the order without using/knowing the exact solution
    # convergence table to determine the behavior of the method for our problem    
    
    with open("behavior_convergence.txt", 'w', encoding='utf-8') as file2:
        file2.write("ORDER BEHAVIOR CONVERGENCE TABLE\n");
    
        e=p=q=r=s1=s2=0;
        for i in range(1,m+1):
            n=16*2**(i-1); 
            if i>2:
                z3= np.array(yn[i-3])
                z2 = np.array(yn[i-2])
                z1 = np.array(yn[i-1])
                s1 = ((z3 - z2)[0])**2 +((z3 - z2)[1])**2
                s2 = ((z2 - z1)[0])**2 +((z2 - z1)[1])**2
                q = s1/s2;
                r = h[i-2]/h[i-1];
            
                p = math.log(q)/math.log(r);
            
                e = (z2-z1)[0]**2 + (z2-z1)[1]**2;
                print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n,h[i-1],e,p)); 
                file2.write("{:5d} & {:9.3e} & {:9.3e} & {:9.3e}\\\\\n".format(n,h[i-1],e,p))   
    
############################################################################
    
main()

