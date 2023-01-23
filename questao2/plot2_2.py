
# MAP5725
# based on Roma's program.

# plots for general explicit one-Step methods.

# 2- (manufactured) problem with kwnown exact solution 
#            (1) I1' = -4I1 + 3I2 + 6, I1(0) = 0
#            (2) I2' = -2.4I1 + 1.6I2 + 3.6, I2(0) = 0
                         

import matplotlib.pyplot as plt
import numpy as np

#############################################################################

def phi(t,y,f):
    # define discretization function 
    return f(t,y) # euler method

############################################################################

def f(t, y):
   # bidimensional problem
    f0 = -4*y[0] + 3*y[1] + 6
    f1 = -2.4*y[0] + 1.6*y[1] + 3.6
    
    return np.array([f0, f1])

############################################################################

# other relevant data
t_n_1 = [0]; t_n_2 = [0]; T = 1;        # time interval: t in [t0,T]
y_n_1 = [np.array([0, 0])]; y_n_2 = [np.array([0, 0])]; # initial condition

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

plt.plot(t_n_1, y_n_1[:,0], 'k--', label = 'I1(t)  n = 16')
plt.plot(t_n_2, y_n_2[:,0], 'k-', label = 'I1(t)  n = 256')

plt.plot(t_n_1, y_n_1[:,1], 'k:', label = 'I2(t)  n = 16')
plt.plot(t_n_2, y_n_2[:,1], 'k-.', label = 'I2(t)   n = 256')

plt.xlabel('time t   (in t units)')
plt.ylabel('y  state variables')
plt.title('Numerical Approximation of State Variables')
plt.legend()
plt.show()