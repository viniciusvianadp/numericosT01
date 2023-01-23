
# MAP5725
# based on Roma's program.

# plots for general explicit one-Step methods.

# 3- (manufactured) problem with kwnown exact solution 
#              y"-2y'+2y = (e^2t)*sin(t), 0<=t<=1, y(0) = -0.4, y'(0) = -0.6
                         

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
t_n_1 = [0]; t_n_2 = [0]; t_n_3 = [0];  T = 1;        # time interval: t in [t0,T]
y_n_1 = [np.array([-0.4, -0.6])]; y_n_2 = [np.array([-0.4, -0.6])];
y_n_3 = [np.array([-0.4, -0.6])]; # initial condition

n_1 = 16                # time interval partition (discretization)
dt = (T-t_n_1[-1])/n_1
while t_n_1[-1] < T:
    y_n_1.append(y_n_1[-1] + dt*phi(t_n_1[-1],y_n_1[-1],f))
    t_n_1.append(t_n_1[-1] + dt)
    dt = min(dt, T-t_n_1[-1])

y_n_1 = np.array(y_n_1)

n_2 = 64                # time interval partition (discretization)
dt = (T-t_n_2[-1])/n_2
while t_n_2[-1] < T:
    y_n_2.append(y_n_2[-1] + dt*phi(t_n_2[-1],y_n_2[-1],f))
    t_n_2.append(t_n_2[-1] + dt)
    dt = min(dt, T-t_n_2[-1])

y_n_2 = np.array(y_n_2)

n_3 = 128                # time interval partition (discretization)
dt = (T-t_n_3[-1])/n_3
while t_n_3[-1] < T:
    y_n_3.append(y_n_3[-1] + dt*phi(t_n_3[-1],y_n_3[-1],f))
    t_n_3.append(t_n_3[-1] + dt)
    dt = min(dt, T-t_n_3[-1])

y_n_3 = np.array(y_n_3)


## plotting the graphic for y1
plt.plot(t_n_1, y_n_1[:,0], 'k:', label = 'n = 16')
plt.plot(t_n_2, y_n_2[:,0], 'k--', label = 'n = 64')
plt.plot(t_n_3, y_n_3[:,0], 'k-', label = 'n = 128')


plt.xlabel('time t   (in t units)')
plt.ylabel('state variable y1')
plt.title('Numerical Approximation of State Variable y1')
plt.legend()
plt.show()

## plotting the graphic for y2
plt.plot(t_n_1, y_n_1[:,1], 'k:', label = 'n = 16')
plt.plot(t_n_2, y_n_2[:,1], 'k--', label = 'n = 64')
plt.plot(t_n_3, y_n_3[:,1], 'k-', label = 'n = 128')


plt.xlabel('time t   (in t units)')
plt.ylabel('state variable y2')
plt.title('Numerical Approximation of State Variable y2')
plt.legend()
plt.show()


## exact vs approximated (y1)
t = np.linspace(0, 1, 128)
plt.plot(t, 0.2*np.exp(2*t)*(np.sin(t) - 2*np.cos(t)), 'k-', label = 'exact')
plt.plot(t_n_3, y_n_3[:, 0], 'k--', label = 'approximated')

plt.xlabel('time t  (in t units)')
plt.ylabel('y1(t)  (in y1 units)')
plt.title('Comparison between approximated and exact solution')
plt.legend()
plt.show()

## exact vs approximated (y2)
t = np.linspace(0, 1, 128)
plt.plot(t, 0.8*np.exp(2*t)*np.sin(t) - 0.6*np.exp(2*t)*np.cos(t), 'k-', label = 'exact')
plt.plot(t_n_3, y_n_3[:, 1], 'k--', label = 'approximated')

plt.xlabel('time t  (in t units)')
plt.ylabel('y2(t)  (in y2 units)')
plt.title('Comparison between approximated and exact solution')
plt.legend()
plt.show()


