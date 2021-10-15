import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np

colores = [(1,1,1), (0,0,1), (0,1,0),(1,0,0)]  # White->B->G->R
bounds = np.array([0., 1., 5., 10., 15., 20., 25., 30., 40., 50., 60., 70., 80., 90., 100., 150., 1000.])
cMap = colors.LinearSegmentedColormap.from_list('mi_precip', colores, N=len(bounds)-1)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)
tick_l = ['0', '1', '5', '10', '15', '20', '25', '30', '40', '50', '60', '70', '80', '90', '100', '150', '']

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])

cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=cMap,
                               norm=norm,  # vmax and vmin
                               ticks=bounds,
                               drawedges=True)
cb.ax.yaxis.set_ticks_position('left')
#cb.set_label(label=u'Precipitación (mm)', weight='bold')
cb.ax.set_title(u'PP (mm)', weight='bold')
cb.ax.set_yticklabels(tick_l, weight='bold', fontsize=9)
plt.savefig('just_colorbar', bbox_inches='tight')
