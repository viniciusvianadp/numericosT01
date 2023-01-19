
# MAP5725
# based on Roma's program.

# plots for general explicit one-Step methods.

# 1- (manufactured) problem with kwnown exact solution 
#              y' = y-t²+1, 0<=t<=2, y(0)=1/2
                         

import matplotlib.pyplot as plt
import numpy as np

#############################################################################

def phi(t,y,f):
    # define discretization function 
    return f(t,y) # euler method

############################################################################

def f(t, y):
    # unidimensional problem
        
    f0 =  y[0] - t**2 + 1
    
    return np.array([f0])

############################################################################

# other relevant data
t_n = [0]; T = 2;        # time interval: t in [t0,T]
y_n = [np.array([0.5])]   # initial condition

n = 10000                  # time interval partition (discretization)
dt = (T-t_n[-1])/n
while t_n[-1] < T:
    y_n.append(y_n[-1] + dt*phi(t_n[-1],y_n[-1],f))
    t_n.append(t_n[-1] + dt)
    dt = min(dt, T-t_n[-1])

y_n = np.array(y_n)

plt.plot(t_n, y_n[:], 'k-', label = 'y(t)  (in y units)')
plt.xlabel('time t   (in t units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()
