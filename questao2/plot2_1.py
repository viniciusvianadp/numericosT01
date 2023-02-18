## 2023.01.23
## Keith Ando Ogawa - keith.ando@usp.br
## Vinícius Viana de Paula - viniciusviana@usp.br

# MAP3122

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
y_n_1 = [np.array([0.5])]; y_n_2 = [np.array([0.5])];
y_n_3 = [np.array([0.5])];  # initial condition

n_1 = 16                 # time interval partition (discretization)
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


## y for different values of n
plt.plot(t_n_1, y_n_1[:], 'k:', label = 'n = 16')
plt.plot(t_n_2, y_n_2[:], 'k--', label = 'n = 64')
plt.plot(t_n_3, y_n_3[:], 'k-', label = 'n = 128')

plt.xlabel('t  (em unidade de tempo)')
plt.ylabel('y(t)  (em unidade de y)')
plt.title('Aproximação Numérica da Variável de Estado y')
plt.legend()
plt.show()

## exact vs approximated
t = np.linspace(0, 2, 128)
plt.plot(t, (t+1)**2 -  0.5*np.exp(t), 'k-', label = 'solução exata')
plt.plot(t_n_3, y_n_3[:], 'k--', label ='solução numérica')

plt.xlabel('t  (em unidade de tempo)')
plt.ylabel('y(t)  (em unidade de y)')
plt.title('Soluções Aproximada e Exata para y')
plt.legend()
plt.show()
