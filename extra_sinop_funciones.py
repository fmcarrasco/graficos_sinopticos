import numpy as np
import netCDF4
import scipy.ndimage as ndimage
import datetime as dt

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.io.shapereader import Reader
from cartopy.io.shapereader import natural_earth
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib import colors as c
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sinop_funciones import mapa_base
from sinop_funciones import get_index_time
from sinop_funciones import get_index_lat
from sinop_funciones import extract_var

def extraer_variable(file, fecha, nomvar, llat, llon):
    """
    Extrae variables en espacio (X, Y) - Tiempo para la variable
    pedida en nomvar
    """
    l_lat = llat
    l_lon = np.array(llon) % 360
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    tiempos = get_index_time(file, fecha)
    # Creamos una variable aux
    res = np.empty([7, len(lat), len(lon)])
    res[:] = np.nan
    fdates = []
    if nomvar == 'precip':
        # Leemos la variable
        ppinit = file.variables['apcpsfc'][:, i_lat[0]:i_lat[1]+1,
                                           i_lon[0]:i_lon[1]+1]
        d0 = tiempos[4]  # --> Initial day at 12UTC (=9 Local Time)

        for dia in np.arange(0, 7):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=9)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            fdates.append(di)
            res[dia,:,:] = np.ma.sum(ppinit[i_t1:i_t2,:,:], axis=0)
    elif nomvar == 'tmax':
        # Leemos la variable
        tmax2m = file.variables['tmax2m'][:,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        d0 = tiempos[1]  # --> Initial day at 03UTC (= 00 Local Time)
        for dia in np.arange(0, 7):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=0)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            fdates.append(di)
            res[dia,:,:] = np.ma.max(tmax2m[i_t1:i_t2,:,:], axis=0)
    elif nomvar == 'tmin':
        tmin2m = file.variables['tmin2m'][:,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        d0 = tiempos[1]  # --> Initial day at 03UTC (= 00 Local Time)
        for dia in np.arange(0, 7):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=0)
            print(di)
            print(di_f)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            res[dia,:,:] = np.ma.min(tmin2m[i_t1:i_t2,:,:], axis=0)
    else:
        print('Solo hay programado para Precip, Tmax y Tmin diaria')
        print('Se devuelve una matriz con NaN')
    ###### End of OPTIONs ##############

    return res, fdates


def plot_precip_daily(file, llat, llon, fecha, carpeta):
    """
    Estimates daily precipitation for each day since start till seven days later
    and plot it.
    """
    # Obtenemos las variables a graficar
    PP, fdates = extraer_variable(file, fecha, 'precip', llat, llon)
    # Shape:(7,Lat,Lon)
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # Datos para el Mapa
    fig1, ax1 = mapa_base(llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#959392', '#5BC5F5',
                             '#E31903', '#7A0B0F'])
    bounds = np.array([0., 1., 20., 50., 100., 150., 1000.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=6)
    for t in np.arange(0, np.shape(PP)[0]):
        z = PP[t, :, :]
        z1 = ndimage.gaussian_filter(z, sigma=1., order=0)
        z1[lndsfc == 0.] = np.nan
        CS = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        cbaxes = inset_axes(ax1, width="50%", height="2%", loc=4)
        cb = plt.colorbar(CS, cax=cbaxes, orientation='horizontal', drawedges=True)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.set_xticklabels(['0', '1', '20', '50', '100', '150', ''],
                              weight='bold', fontsize=13)
        fig_name = carpeta + 'PP_' + fdates[t].strftime('%Y%m%d') + '.png'
        plt.savefig(fig_name, dpi=200)


if __name__ == '__main__':
    import os
    #
    mydate='20190530'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    ofolder = 'e:/python/graficos_sinopticos/' + mydate + '/'
    os.makedirs(ofolder, exist_ok=True)
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p50/gfs' + mydate +\
         '/gfs_0p50_00z'
    file = netCDF4.Dataset(url)
    #extract_variable(file, mydate, 'tmax', l_lat, l_lon)
    plot_precip_daily(file, l_lat, l_lon, mydate, ofolder)
    file.close()
