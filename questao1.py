import matplotlib.pyplot as plt
import numpy as np
import math

t = np.linspace(-math.pi, math.pi, 100)

plt.plot(t, np.cos(t), 'k-', label='cos(t)')
plt.plot(t, np.cos(2*t), 'k--', label='cos(2t)')
plt.plot(t, np.cos(3*t), 'k:', label='cos(3t)')
plt.xlabel("t (adimensional)")
plt.ylabel("y(t) (adimensional)")
plt.title("Funções cos(mt) para m entre 1 e 3")
plt.legend();

plt.show()