import matplotlib.pyplot as plt
import numpy as np

step = np.array([1000, 2000, 3000, 4000, 5000, 6000])
loss = np.array([1.0027, 0.8486, 0.7695, 0.5265, 0.4725, 0.5187])
em = np.array([76.21440536013401, 78.55946398659967, 80.77051926298158, 81.1390284757119, 81.90954773869346, 82.68006700167504])


# Plotting the Graph
fig, axs = plt.subplots(2)
axs[0].plot(step, loss, "-o")

axs[1].plot(step, em, "-o")

for i, ax in enumerate(axs.flat):
    if (i ==0):
        ax.set(xlabel='step', ylabel='loss')
    else:
        ax.set(xlabel='step', ylabel='em')
plt.show()
