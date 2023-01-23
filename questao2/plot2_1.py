
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
t_n_1 = [0]; t_n_2 = [0]; t_n_3 = [0]; T = 2;        # time interval: t in [t0,T]
y_n_1 = [np.array([0.5])]; y_n_2 = [np.array([0.5])]; y_n_3 = [np.array([0.5])];  # initial condition

n_1 = 2                 # time interval partition (discretization)
dt = (T-t_n_1[-1])/n_1
while t_n_1[-1] < T:
    y_n_1.append(y_n_1[-1] + dt*phi(t_n_1[-1],y_n_1[-1],f))
    t_n_1.append(t_n_1[-1] + dt)
    dt = min(dt, T-t_n_1[-1])

y_n_1 = np.array(y_n_1)

n_2 = 16                # time interval partition (discretization)
dt = (T-t_n_2[-1])/n_2
while t_n_2[-1] < T:
    y_n_2.append(y_n_2[-1] + dt*phi(t_n_2[-1],y_n_2[-1],f))
    t_n_2.append(t_n_2[-1] + dt)
    dt = min(dt, T-t_n_2[-1])

y_n_2 = np.array(y_n_2)

n_3 = 256                # time interval partition (discretization)
dt = (T-t_n_3[-1])/n_3
while t_n_3[-1] < T:
    y_n_3.append(y_n_3[-1] + dt*phi(t_n_3[-1],y_n_3[-1],f))
    t_n_3.append(t_n_3[-1] + dt)
    dt = min(dt, T-t_n_3[-1])

y_n_3 = np.array(y_n_3)

plt.plot(t_n_1, y_n_1[:], 'k:', label = 'y(t)  n = 2')
plt.plot(t_n_2, y_n_2[:], 'k--', label = 'y(t)  n = 16')
plt.plot(t_n_3, y_n_3[:], 'k-', label = 'y(t)   n = 256')

plt.xlabel('time t   (in t units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()
