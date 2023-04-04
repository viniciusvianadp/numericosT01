## 2023.01.23

import matplotlib.pyplot as plt ## to plot the graphics
import numpy as np ## to use mathematical resources

t = np.linspace(-np.pi, np.pi, 100)

plt.plot(t, np.cos(t), 'k-', label='cos(t)')
plt.plot(t, np.cos(2*t), 'k--', label='cos(2t)')
plt.plot(t, np.cos(3*t), 'k:', label='cos(3t)')

plt.xlabel("t (adimensional)")
plt.ylabel("y(t) (adimensional)")

plt.title("Funções cos(mt) para m entre 1 e 3")
plt.legend()

plt.show()
