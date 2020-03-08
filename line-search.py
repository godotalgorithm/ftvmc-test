import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x1, e1 = np.loadtxt('test10.dat', usecols=(1, 2), unpack=True)
x2, e2 = np.loadtxt('test100.dat', usecols=(1, 2), unpack=True)
x3, e3 = np.loadtxt('test1000.dat', usecols=(1, 2), unpack=True)
x4, e4 = np.loadtxt('test10000.dat', usecols=(1, 2), unpack=True)

# plot the error trends
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(6.0,3.0))
ax1.set_xlim([0,1])
ax1.set_ylim([-2.38,-2.18])
ax2.set_xlim([0.8,1])
ax2.set_ylim([-2.325,-2.315])

options = {'lw':0.75}
ax1.plot(x1,e1,**options)
ax1.plot(x2,e2,**options)
ax1.plot(x3,e3,**options)
ax1.plot(x4,e4,**options)
ax1.plot([0,1],[-2.192858,-2.322403],'ko',ms=3,clip_on=False)
ax1.text(0.87, -2.37, '10',size='smaller')

rect = patches.Rectangle((0.8,-2.325),0.2,0.01,linewidth=0.75,edgecolor='k',facecolor='none',ls='--')
ax1.add_patch(rect)

ax2.plot(x1,e1,**options)
ax2.plot(x2,e2,**options)
ax2.plot(x3,e3,**options)
ax2.plot(x4,e4,**options)
ax2.plot([1],[-2.322403],'ko',ms=3,clip_on=False)
ax2.text(0.95, -2.3175, '100',size='smaller')
ax2.text(0.95, -2.3248, '1000',size='smaller')
ax2.text(0.95, -2.321, '10000',size='smaller')

ax1.set_xlabel('x')
ax1.set_ylabel('free energy per site')
plt.savefig('line-search.pdf', bbox_inches='tight', pad_inches=0.01)

