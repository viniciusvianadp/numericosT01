
# MAP5725
# based on Roma's program.

# plots for general explicit one-Step methods.

# 3- (manufactured) problem with kwnown exact solution 
#              y"-2y'+2y = (e^t)*sin(t), 0<=t<=1, y(0) = -0.4, y'(0) = -0.6
                         

import matplotlib.pyplot as plt
import numpy as np
import math

#############################################################################

def phi(t,y,f):
    # define discretization function 
    return f(t,y) # euler method

############################################################################

def f(t, y):
   # bidimensional problem
    f0 =  y[1]
    f1 =  math.exp(2*t)*math.sin(t) - 2*y[0] + 2*y[1]
    
    return np.array([f0, f1])

############################################################################

# other relevant data
t_n_1 = [0]; t_n_2 = [0]; T = 1;        # time interval: t in [t0,T]
y_n_1 = [np.array([-0.4, -0.6])]; y_n_2 = [np.array([-0.4, -0.6])];  # initial condition

n_1 = 16                # time interval partition (discretization)
dt = (T-t_n_1[-1])/n_1
while t_n_1[-1] < T:
    y_n_1.append(y_n_1[-1] + dt*phi(t_n_1[-1],y_n_1[-1],f))
    t_n_1.append(t_n_1[-1] + dt)
    dt = min(dt, T-t_n_1[-1])

y_n_1 = np.array(y_n_1)

n_2 = 256                # time interval partition (discretization)
dt = (T-t_n_2[-1])/n_2
while t_n_2[-1] < T:
    y_n_2.append(y_n_2[-1] + dt*phi(t_n_2[-1],y_n_2[-1],f))
    t_n_2.append(t_n_2[-1] + dt)
    dt = min(dt, T-t_n_2[-1])

y_n_2 = np.array(y_n_2)

plt.plot(t_n_1, y_n_1[:,0], 'k--', label = 'y1(t)  n = 16')
plt.plot(t_n_2, y_n_2[:,0], 'k-', label = 'y1(t)  n = 256')

plt.plot(t_n_1, y_n_1[:,1], 'k:', label = 'y2(t)  n = 16')
plt.plot(t_n_2, y_n_2[:,1], 'k-.', label = 'y2(t)   n = 256')

plt.xlabel('time t   (in t units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()
