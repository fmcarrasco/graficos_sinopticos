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
            res[dia, :, :] = ppinit[i_t2, :, :] - ppinit[i_t1, :, :]
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
            res[dia, :, :] = np.ma.max(tmax2m[i_t1:i_t2, :, :], axis=0)
    elif nomvar == 'tmin':
        tmin2m = file.variables['tmin2m'][:,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        d0 = tiempos[1]  # --> Initial day at 03UTC (= 00 Local Time)
        for dia in np.arange(0, 7):
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


def extract_wind(file, fecha, llat, llon):
    """
    """
    l_lat = llat
    l_lon = np.array(llon) % 360
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    tiempos = get_index_time(file, fecha)
    di = tiempos[1]
    di_f = (di + dt.timedelta(days=7)).replace(hour=0)
    i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
    i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
    # Creamos una variable aux
    res = np.empty([i_t2 - i_t1 + 1, len(lat), len(lon)])
    res[:] = np.nan
    fdates = [datef for datef in tiempos if datef>=di and datef<=di_f]
    u10m = file.variables['ugrd10m'][i_t1:i_t2, i_lat[0]:i_lat[1]+1,
                                     i_lon[0]:i_lon[1]+1]
    v10m = file.variables['vgrd10m'][i_t1:i_t2, i_lat[0]:i_lat[1]+1,
                                     i_lon[0]:i_lon[1]+1]

    return lon, lat, u10m, v10m, fdates


def extract_pphour(file, fecha, llat, llon):
    """
    """
    l_lat = llat
    l_lon = np.array(llon) % 360
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    tiempos = get_index_time(file, fecha)
    di = tiempos[1]
    di_f = (di + dt.timedelta(days=7)).replace(hour=0)
    i_t1 = [i for i in range(len(tiempos)) if tiempos[i] == di][0]
    i_t2 = [i for i in range(len(tiempos)) if tiempos[i] == di_f][0]
    # Creamos una variable aux
    res = np.empty([i_t2 - i_t1 + 1, len(lat), len(lon)])
    res[:] = np.nan
    fdates = [datef for datef in tiempos if datef>=di and datef<=di_f]
    pp1 = file.variables['apcpsfc'][i_t1:i_t2, i_lat[0]:i_lat[1]+1,
                                    i_lon[0]:i_lon[1]+1]
    pp2 = file.variables['apcpsfc'][i_t1+1:i_t2+1, i_lat[0]:i_lat[1]+1,
                                    i_lon[0]:i_lon[1]+1]
    pphour = pp2 - pp1

    return lon, lat, pphour, fdates


def plot_precip_daily(file, llat, llon, fecha, prefix):
    """
    Estimates daily precipitation for each day since start till seven days later
    and plot it.
    """
    # Obtenemos las variables a graficar
    PP, fdates = extraer_variable(file, fecha, 'precip', llat, llon)
    # Shape:(7,Lat,Lon)
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#959392', '#5BC5F5',
                             '#E31903', '#7A0B0F'])
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#66c2a4', '#959392',
                             '#5BC5F5', '#E31903', '#7A0B0F'])
    bounds = np.array([0., 1., 10., 20., 50., 100., 150., 1000.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=7)
    for t in np.arange(0, np.shape(PP)[0]):
        z = PP[t, :, :]
        z1 = ndimage.gaussian_filter(z, sigma=1., order=0)
        z1[lndsfc == 0.] = np.nan
        # Datos para el Mapa
        fig1, ax1 = mapa_base(llat, llon)
        CS = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        cbaxes = inset_axes(ax1, width="50%", height="2%", loc=4)
        cb = plt.colorbar(CS, cax=cbaxes, orientation='horizontal', drawedges=True)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.set_xticklabels(['0', '1', '10', '20', '50', '100', '150', ''],
                              weight='bold', fontsize=13)
        fig_name = prefix + fdates[t].strftime('%Y%m%d') + '.png'
        plt.savefig(fig_name, dpi=200)
        ax1.clear()


def plot_temp_daily(file, llat, llon, fecha, prefix):
    """
    Estimates daily precipitation for each day since start till seven days later
    and plot it.
    """
    # Obtenemos las variables a graficar
    TMAX, fdates = extraer_variable(file, fecha, 'tmax', llat, llon)
    TMIN, fdates = extraer_variable(file, fecha, 'tmin', llat, llon)
    # Shape:(7,Lat,Lon)
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    bounds = np.arange(-18, 46, 2)
    bounds[0] = -50
    bounds[-1] = 60
    cMap = c.ListedColormap(['#3A027E', '#480092', '#06187B', '#162892',
                             '#203CB7', '#2A50C1', '#3066DB', '#417AF8',
                             '#508CF9', '#619FF8', '#76B2F2', '#87C6FD',
                             '#93DEFE', '#ACECF7', '#C7F4F8', '#93E4B0',
                             '#A5F789', '#D1FC7A', '#E7EE90', '#FCE5A1',
                             '#FBCB61', '#FBB06D', '#FF7B40', '#FC5C2A',
                             '#FF3A1D', '#DE0406', '#B80004', '#960103',
                             '#6B0207', '#450102', '#250400'])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=31)
    for t in np.arange(0, np.shape(TMAX)[0]):
        z1 = TMAX[t, :, :] - 273.
        z1[lndsfc == 0.] = np.nan
        z2 = TMIN[t, :, :] - 273.
        z2[lndsfc == 0.] = np.nan
        # Datos para el Mapa
        fig1, ax1 = mapa_base(llat, llon)
        fig2, ax2 = mapa_base(llat, llon)
        # Tmax
        CS0 = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        CS1 = ax1.contour(x, y, z1, levels=[35., 40., 100.],
                          colors=['#000000', '#000000'],
                          linestyles=['solid', 'dashed'],
                          linewidths=[0.7, 0.7],
                          transform=ccrs.PlateCarree())
        cbaxes = inset_axes(ax1, width="3%", height="50%", loc=4)
        cb = fig1.colorbar(CS0, cax=cbaxes, drawedges=True, ticks=bounds)
        cb.ax.yaxis.set_ticks_position('left')
        for label in cb.ax.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        cb.ax.yaxis.get_ticklabels()[-1].set_visible(False)
        fig_name = prefix[0] + fdates[t].strftime('%Y%m%d') + '.png'
        fig1.savefig(fig_name, dpi=200)
        # Tmin
        CS2 = ax2.contourf(x, y, z2, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        CS3 = ax2.contour(x, y, z2, levels=[-100., 0., 3.],
                          colors=['#000000', '#000000'],
                          linestyles=['dashed', 'solid'],
                          linewidths=[0.7, 0.7],
                          transform=ccrs.PlateCarree())
        cbaxes1 = inset_axes(ax2, width="3%", height="50%", loc=4)
        cb1 = fig1.colorbar(CS2, cax=cbaxes1, drawedges=True, ticks=bounds)
        cb1.ax.yaxis.set_ticks_position('left')
        for label in cb1.ax.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        cb.ax.yaxis.get_ticklabels()[-1].set_visible(False)
        fig_name = prefix[1] + fdates[t].strftime('%Y%m%d') + '.png'
        fig2.savefig(fig_name, dpi=200)
        ax1.clear()
        ax2.clear()


def plot_wind_daily(file, llat, llon, fecha, prefix):
    """
    """
    lon, lat, u, v, fdates = extract_wind(file, fecha, llat, llon)
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    wspd = np.sqrt(np.power(u, 2) + np.power(v, 2))
    cMap = c.ListedColormap(['#ffffff', '#b3cde3', '#8c96c6', '#8856a7',
                             '#810f7c'])
    bounds = [0., 10., 13., 16., 20., 50.]
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=5)
    skip=(slice(None,2), slice(None,2))
    for t in np.arange(0, np.shape(u)[0]):
        ws10 = wspd[t, :, :]
        u10 = u[t, :, :]/ws10
        v10 = v[t, :, :]/ws10
        # Datos Mapa
        fig, ax = mapa_base(llat, llon)
        CS = ax.contourf(x, y, ws10, levels=bounds, cmap=cMap, norm=norm,
                         transform=ccrs.PlateCarree())
        QW = ax.quiver(x[::3], y[::3], u10[::3, ::3], v10[::3, ::3],
                       transform=ccrs.PlateCarree())
        cbaxes = inset_axes(ax, width="3%", height="30%", loc=4)
        cb = fig.colorbar(CS, cax=cbaxes, drawedges=True, ticks=bounds[0:-1])
        cb.ax.yaxis.set_ticks_position('left')
        fig_name = prefix + fdates[t].strftime('%Y%m%d%H') + '.png'
        fig.savefig(fig_name, dpi=200)
        ax.clear()


def plot_precip_hourly(file, llat, llon, fecha, prefix):
    """
    Estimates daily precipitation for each day since start till seven days later
    and plot it.
    """
    # Obtenemos las variables a graficar
    lon, lat, pphour, fdates = extract_pphour(file, fecha, llat, llon)
    # Shape:(7,Lat,Lon)
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    # Datos para el Mapa
    fig1, ax1 = mapa_base(llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#66c2a4', '#959392',
                             '#5BC5F5', '#E31903', '#7A0B0F'])
    bounds = np.array([0., 1., 10., 20., 50., 100., 150., 1000.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=7)
    for t in np.arange(0, np.shape(pphour)[0]):
        z = pphour[t, :, :]
        z1 = ndimage.gaussian_filter(z, sigma=1., order=0)
        z1[lndsfc == 0.] = np.nan
        CS = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                           transform=ccrs.PlateCarree())
        cbaxes = inset_axes(ax1, width="50%", height="2%", loc=4)
        cb = plt.colorbar(CS, cax=cbaxes, orientation='horizontal', drawedges=True)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.set_xticklabels(['0', '1', '10', '20', '50', '100', '150', ''],
                              weight='bold', fontsize=13)
        fig_name = prefix + fdates[t].strftime('%Y%m%d%H') + '.png'
        plt.savefig(fig_name, dpi=200)


if __name__ == '__main__':
    import os
    #
    mydate='20190715'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    ofolder1 = 'e:/python/graficos_sinopticos/' + mydate + '_05deg/'
    os.makedirs(ofolder1, exist_ok=True)
    ofolder2 = 'e:/python/graficos_sinopticos/' + mydate + '_025deg/'
    os.makedirs(ofolder2, exist_ok=True)
    # 05 deg
    print('--- Graficando GFS con 0.5 grados de resolucion ---')
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p50/gfs' + mydate +\
         '/gfs_0p50_00z'
    file = netCDF4.Dataset(url)
    #
    prefijo = ofolder1 + 'PP_'
    plot_precip_daily(file, l_lat, l_lon, mydate, prefijo)
    prefijos = [ofolder1 + 'TMAX_', ofolder1 + 'TMIN_']
    plot_temp_daily(file, l_lat, l_lon, mydate, prefijos)
    prefijo = ofolder1 + 'SurfWind_'
    plot_wind_daily(file, l_lat, l_lon, mydate, prefijo)
    # prefijo = ofolder1 + 'hora_PP_'
    # plot_precip_hourly(file, l_lat, l_lon, mydate, prefijo)
    file.close()
    # 0.25 deg
    print('--- Graficando GFS con 0.25 grados de resolucion ---')
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs' + mydate +\
         '/gfs_0p25_00z'
    file = netCDF4.Dataset(url)
    #
    prefijo = ofolder2 + 'PP_'
    plot_precip_daily(file, l_lat, l_lon, mydate, prefijo)
    prefijos = [ofolder2 + 'TMAX_', ofolder2 + 'TMIN_']
    plot_temp_daily(file, l_lat, l_lon, mydate, prefijos)
    prefijo = ofolder2 + 'SurfWind_'
    plot_wind_daily(file, l_lat, l_lon, mydate, prefijo)
    # prefijo = ofolder2 + 'hora_PP_'
    # plot_precip_hourly(file, l_lat, l_lon, mydate, prefijo)
    file.close()
