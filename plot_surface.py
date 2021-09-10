import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd

data = pd.read_excel('Sim Results.xlsx', sheet_name='Muons per proton')

print(data)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
thickness = data.iloc[0, 1:]
energy = data.iloc[1:, 0]
muons_per_proton = data.iloc[1:, 1:]

thickness_array = np.array(thickness)
energy_array = np.array(energy)
muons_per_proton_array = np.array(muons_per_proton)

X, Y = np.meshgrid(thickness_array, energy_array)

# Plot the surface.
surf = ax.plot_surface(X, Y, muons_per_proton_array, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Target thickness (mm)')
ax.set_ylabel('Beam energy (MeV)')
ax.set_zlabel('Muons per attenuated proton')
# plt.title('Proton beam attenuation as a function of muon target thickness and proton beam energy')

'''
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
'''

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)

plt.show()