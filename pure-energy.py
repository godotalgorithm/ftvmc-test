import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

h1, exact1, ref1 = np.loadtxt('pure-energy0.dat', usecols=(0, 1, 2), unpack=True)
h2, ref2, ea2, eb2, barb2, ec2, barc2, ed2, bard2 = np.loadtxt('pure-energy.dat', usecols=(0, 2, 3, 4, 5, 6, 7, 8, 9), unpack=True)

# plot the error trends
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(6.0,3.0))
ax1.set_xlim([0,2])
ax1.set_ylim([-0.05,0.15])
ax2.set_xlim([0,2])
ax2.set_ylim([-0.025,0.005])

options = {'lw':0.75,'color':'black'}
ax1.plot(h1,exact1-ref1,**options)
ax1.plot(h2,ea2-ref2,'ro',ms=1.5,label=r'$n=0$',clip_on=False)
ax1.errorbar(h2,eb2-ref2,yerr=barb2,fmt='g^',ms=1.5,lw=0.75,label=r'$n=1$',clip_on=False)
ax1.errorbar(h2,ec2-ref2,yerr=barc2,fmt='bs',ms=1.5,lw=0.75,label=r'$n=2$',clip_on=False)
ax1.errorbar(h2,ed2-ref2,yerr=bard2,fmt='D',color='orange',ms=1.5,lw=0.75,label=r'$n=3$',clip_on=False)

rect = patches.Rectangle((0.0,-0.025),2.0,0.03,linewidth=0.75,edgecolor='k',facecolor='none',ls='--')
ax1.add_patch(rect)
ax1.legend()

ax2.plot(h1,exact1-ref1,**options)
ax2.plot(h2,ea2-ref2,'ro',ms=1.5)
ax2.errorbar(h2,eb2-ref2,yerr=barb2,fmt='g^',ms=1.5,lw=0.75,clip_on=False)
ax2.errorbar(h2,ec2-ref2,yerr=barc2,fmt='bs',ms=1.5,lw=0.75,clip_on=False)
ax2.errorbar(h2,ed2-ref2,yerr=bard2,fmt='D',color='orange',ms=1.5,lw=0.75,clip_on=False)

ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel('residual energy per site')
plt.savefig('pure-energy.pdf', bbox_inches='tight', pad_inches=0.01)

