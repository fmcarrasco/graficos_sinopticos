#
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
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def mapa_base(llat, llon):
    """
    Mapa base para graficar las variables
    """

    l_lat = llat
    l_lon = np.array(llon) % 360  #Pasamos lon en [-180, 180] a [0, 360]
    states_provinces = cpf.NaturalEarthFeature(category='cultural',
                            name='admin_1_states_provinces_lines',
                            scale='10m',
                            facecolor='none')
    shp = Reader(natural_earth(resolution='10m', category='cultural',
                               name='admin_1_states_provinces_lines'))
    countries = shp.records()

    # Comenzamos la Figura
    fig = plt.figure(figsize=(6, 8))
    proj_lcc = ccrs.PlateCarree()
    ax = plt.axes(projection=proj_lcc)
    ax.coastlines(resolution='10m')
    ax.add_feature(cpf.BORDERS, linestyle='-')
    #ax.add_feature(states_provinces, edgecolor='gray')
    for country in countries:
        if country.attributes['adm0_name'] == 'Argentina':
            ax.add_geometries( [country.geometry], ccrs.PlateCarree(),
                                edgecolor='black', facecolor='none',
                                linewidth=0.7 )
    # ax.add_feature(shape_feature)
    # Colocamos reticula personalizada
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='gray', alpha=0.7, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.linspace(llon[0], llon[1], 7))
    gl.ylocator = mticker.FixedLocator(np.linspace(l_lat[0], l_lat[1], 9))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # Extension del mapa
    ax.set_extent([l_lon[0], l_lon[1], l_lat[0], l_lat[1]], crs=proj_lcc)
    # Posicion del eje (desplazamos un poco a la izquierda y más abajo)
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 - 0.05, pos1.y0 - 0.06,  pos1.width*1.16, pos1.height*1.22]
    ax.set_position(pos2) # set a new position
    ax.text(-79., -21., 'Fuente:\n NOAA - GFS',
            horizontalalignment='left', verticalalignment='top',
            fontweight='bold', fontsize=13,
            transform=ccrs.Geodetic())
    return fig, ax


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
    ndays = 8
    res = np.empty([ndays, len(lat), len(lon)])
    res[:] = np.nan
    fdates = []
    if nomvar == 'precip':
        # Leemos la variable
        ppinit = file.variables['apcpsfc'][:, i_lat[0]:i_lat[1]+1,
                                           i_lon[0]:i_lon[1]+1]
        i1 = np.min(np.where(np.array([a.hour for a in tiempos])==12))
        # primer tiempo que inicia a las 12Z
        d0 = tiempos[i1]  # --> Initial day at 12UTC (=9 Local Time)
        for dia in np.arange(0, ndays):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=9)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            fdates.append(di)
            res[dia, :, :] = ppinit[i_t2, :, :] - ppinit[i_t1, :, :]
    elif nomvar == 'tmax':
        # Leemos la variable
        tmax2m = file.variables['tmax2m'][:,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        d0 = tiempos[1]  # --> Initial day at 03UTC (= 00 Local Time)
        for dia in np.arange(0, ndays):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=0)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            fdates.append(di)
            res[dia, :, :] = np.ma.max(tmax2m[i_t1:i_t2, :, :], axis=0)
    elif nomvar == 'tmin':
        tmin2m = file.variables['tmin2m'][:,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        d0 = tiempos[1]  # --> Initial day at 03UTC (= 00 Local Time)
        for dia in np.arange(0, ndays):
            di = d0 + dt.timedelta(days=int(dia))
            di_f = (di + dt.timedelta(days=1)).replace(hour=0)
            i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
            i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
            fdates.append(di)
            res[dia, :, :] = np.ma.min(tmin2m[i_t1:i_t2, :, :], axis=0)
    else:
        print('Solo hay programado para Precip, Tmax y Tmin diaria')
        print('Se devuelve una matriz con NaN')
    ###### End of OPTIONs ##############

    return res, fdates


def get_index_lat(fecha, file, llat, llon):
    """
    Get the index values for lat and lon to extract in requested square of data.
    """

    l_lat = llat
    l_lon = np.array(llon) % 360
    flat  = file.variables['lat'][:]
    flon  = file.variables['lon'][:]

    lat = [a for a in flat if (a >= l_lat[0] and a <= l_lat[1])]
    i_lat = [np.where(flat == l_lat[0])[0][0], np.where(flat == l_lat[1])[0][0]]
    lon = [a for a in flon if (a >= l_lon[0] and a <= l_lon[1])]
    i_lon = [np.where(flon == l_lon[0])[0][0], np.where(flon == l_lon[1])[0][0]]

    del(flat)
    del(flon)

    return i_lat, i_lon, lat, lon

def get_index_time(file, fecha):
    """
    """
    aux = file.variables['time'][:]
    a = file.variables['time'].getncattr('units')
    c = netCDF4.num2date(aux, units=a)
    tiempo = [dt.datetime.strptime(str(tp), '%Y-%m-%d %H:%M:%S') for tp in c]

    return tiempo

def cmap_for_precip():
    colores = [(1,1,1), (0,0,1), (0,1,0),(1,0,0)]  # B->G->R
    bounds = np.array([0., 1., 5.,10., 15., 20., 25., 30., 40., 50., 60., 70., 80., 90., 100., 150., 1000.])
    cMap = colors.LinearSegmentedColormap.from_list('mi_precip', colores, N=len(bounds)-1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)
    tick_l = ['0', '5', '15', '25', '40', '60', '80', '100', '']

    return cMap, bounds, norm, tick_l

def plot_precip_daily(file, llat, llon, fecha, prefix):
    """
    Estimates daily precipitation for each day since start till seven days later
    and plot it.
    """
    import locale
    locale.setlocale(locale.LC_ALL, 'esp_esp')
    # Obtenemos las variables a graficar
    PP, fdates = extraer_variable(file, fecha, 'precip', llat, llon)
    # Shape:(7,Lat,Lon)
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    cMap, bounds, norm, tick_l = cmap_for_precip()
    for t in np.arange(0, np.shape(PP)[0]):
        z = PP[t, :, :]
        z1 = ndimage.gaussian_filter(z, sigma=1., order=0)
        # Datos para el Mapa
        fecha_p1 = fdates[t].strftime('%A %d/%m')
        fig1, ax1 = mapa_base(llat, llon)
        CS = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        ax1.text(-79., -24., fecha_p1,
                horizontalalignment='left', verticalalignment='top',
                fontweight='bold', fontsize=13,
                transform=ccrs.Geodetic())
        fig_name = prefix + fdates[t].strftime('%Y%m%d') + '.png'
        plt.savefig(fig_name, dpi=200)
        plt.close(fig1)


if __name__ == '__main__':
    import os
    #
    mydate = '20211015'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    ofolder = 'd:/python/graficos_sinopticos/fecovita_' + mydate + '_025deg/'
    os.makedirs(ofolder, exist_ok=True)
    # 0.25 deg
    print('--- Graficando GFS con 0.25 grados de resolucion ---')
    url ='https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs' + mydate +\
         '/gfs_0p25_12z'
    file = netCDF4.Dataset(url)
    #
    prefijo = ofolder + 'PP_'
    plot_precip_daily(file, l_lat, l_lon, mydate, prefijo)
    file.close()
