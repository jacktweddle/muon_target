import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(filename):
    data = pd.read_csv(filepath_or_buffer=filename,
                       sep=' ', names=['x', 'y', 'z', 'Px', 'Py', 'Pz', 't', 'PDGid',
                                       'EventID', 'TrackID', 'ParentID', 'Weight', 'Bx',
                                       'By', 'Bz', 'Ex', 'Ey', 'Ez', 'ProperTime', 'PathLength',
                                       'PolX', 'PolY', 'PolZ', 'InitX', 'InitY', 'InitZ', 'InitT',
                                       'InitKE'], index_col=False, skiprows=2)
    return data


def angle_calc(x, y, z, x_ini, y_ini, z_ini):
    theta = np.arccos((z-z_ini)/np.sqrt((x-x_ini)**2+(y-y_ini)**2+(z-z_ini)**2))
    phi = np.arctan((y-y_ini)/(x-x_ini))

    if (x-x_ini) < 0:
        phi += np.pi
    if (y-y_ini < 0) & (x-x_ini > 0):
        phi += 2*np.pi

    return theta, phi


detectors = ['face1.txt', 'face2.txt', 'face3.txt', 'face4.txt', 'face5.txt', 'face6.txt', 'proton_det.txt']

muon_plus_count = 0
muon_minus_count = 0
pion_plus_count = 0
pion_minus_count = 0
proton_count = 0
front_proton_count = 0
sec_proton_count = 0
neutron_count = 0
pion_theta = []
pion_phi = []
muon_theta = []
muon_phi = []

for i in range(0, len(detectors)):
    data = read_csv(detectors[i])
    for j in range(0, len(data.PDGid)):
        if detectors[i] == 'proton_det.txt':
            if data.PDGid[j] == 2212 and data.TrackID[j] < 1000:
                front_proton_count += 1
        if data.PDGid[j] == 2212 and data.TrackID[j] >= 1000:
            sec_proton_count += 1
        if data.PDGid[j] == 2112:
            neutron_count += 1
        if data.PDGid[j] in [211, -211]:
            if data.PDGid[j] == 211:
                pion_plus_count += 1
            if data.PDGid[j] == -211:
                pion_minus_count += 1
            theta, phi = angle_calc(data.x[j], data.y[j], data.z[j], data.InitX[j],
                                    data.InitY[j], data.InitZ[j])
            pion_theta.append(theta)
            pion_phi.append(phi)
        if data.PDGid[j] in [13, -13]:
            if data.PDGid[j] == 13:
                muon_minus_count += 1
            if data.PDGid[j] == -13:
                muon_plus_count += 1
            theta, phi = angle_calc(data.x[j], data.y[j], data.z[j], data.InitX[j],
                                    data.InitY[j], data.InitZ[j])
            muon_theta.append(theta)
            muon_phi.append(phi)

print('Front proton count:', front_proton_count)
print('Secondary proton count:', sec_proton_count)
print('Neutron count:', neutron_count)
print('Pion- count:', pion_minus_count)
print('Pion+ count:', pion_plus_count)
print('Muon- count:', muon_minus_count)
print('Muon+ count:', muon_plus_count)

rbins = np.linspace(0, np.pi, 50)
abins = np.linspace(0, 2*np.pi, 50)

hist, _, _ = np.histogram2d(pion_theta, pion_phi, bins=(rbins, abins))
R, A = np.meshgrid(rbins, abins)

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
xT=plt.xticks()[0]
xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL)
plt.xlim(0, np.pi)

pc = ax.pcolormesh(R, A, hist.T, cmap='magma_r')
cbar = fig.colorbar(pc)
cbar.set_label('Number of pions')
ax.grid(True)
plt.show()

rbins = np.linspace(0, np.pi, 20)
abins = np.linspace(0, 2*np.pi, 20)

hist, _, _ = np.histogram2d(muon_theta, muon_phi, bins=(rbins, abins))
R, A = np.meshgrid(rbins, abins)

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
xT=plt.xticks()[0]
xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL)
plt.xlim(0, np.pi)

pc = ax.pcolormesh(R, A, hist.T, cmap='magma_r')
cbar = fig.colorbar(pc)
cbar.set_label('Number of muons')
ax.grid(True)
plt.show()

'''
plt.scatter(pion_theta, pion_phi, s=1)
plt.scatter(muon_theta, muon_phi, s=5)
plt.xlabel('Theta')
plt.ylabel('Phi')
plt.legend(['Pions', 'Muons'])
'''