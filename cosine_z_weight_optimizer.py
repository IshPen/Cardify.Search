# prompt: make overlapping graphs on the domain [0,1] where y=x^z and z is a range of integer from [0, 100]
# note: x^8 and 12 look pretty :P

import numpy as np
import matplotlib.pyplot as plt

# Create data for the plot
x = np.linspace(0, 1, 100)  # 100 points evenly spaced between 0 and 1

# Create the plot
for z in range(21):
  y = x ** z
  plt.plot(x, y, label=f'z = {z}')

plt.xlabel('x')
plt.ylabel('y = x^z')
plt.title('Overlapping Graphs of y = x^z on [0, 1]')
plt.grid(True)
plt.legend()
plt.show()

