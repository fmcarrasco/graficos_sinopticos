# Funciones para graficar mapas sinopticos
# para el pronóstico del informe ORA

import numpy as np
import netCDF4

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors as c

def mapa_base(llat, llon):
    """
    Mapa base para graficar las variables
    """

    l_lat = llat
    l_lon = np.array(llon) % 360
    # Trabajamos con el SHAPEFILE de IGN para provincias
    fname = './shapefile/Provincias'
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                   ccrs.PlateCarree(), edgecolor='black',
                                   facecolor='None', linewidth=0.5)
    # Comenzamos la Figura
    fig = plt.figure(figsize=(6, 8))
    proj_lcc = ccrs.PlateCarree()
    ax = plt.axes(projection=proj_lcc)
    ax.coastlines(resolution='10m')
    ax.add_feature(cpf.BORDERS, linestyle='-')
    ax.add_feature(shape_feature)
    # Colocamos reticula personalizada
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='gray', alpha=0.7, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.linspace(llon[0], llon[1], 7))
    gl.ylocator = mticker.FixedLocator(np.linspace(l_lat[0], l_lat[1], 9))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # Extension del grafico
    ax.set_extent([l_lon[0], l_lon[1], l_lat[0], l_lat[1]], crs=proj_lcc)
    # Posicion del eje (desplazamos un poco a la izquierda y más abajo)
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 - 0.05, pos1.y0 - 0.06,  pos1.width*1.1, pos1.height*1.15]
    ax.set_position(pos2) # set a new position

    return fig, ax

def mapa_tmax(llat, llon, fecha):
    """
    """
    fig1, ax1 = mapa_base(llat, llon)
    tmax = extract_var(fecha, llat, llon, 'tmax')
    i_lat, i_lon, lat, lon = get_index_lat(fecha,llat,llon)
    # Datos para el Mapa
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(tmax) - 273.
    # Ploteamos el Mapa
    cMap = c.ListedColormap(['#ffffff', '#FF6002', '#a70101'])
    bounds = np.array([-10., 35., 40., 100.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=3)
    CS = ax1.contourf(x, y, z, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    figure_name = 'tmax_' + fecha + '.png'
    plt.savefig(figure_name, dpi=200)

def mapa_tmin(llat, llon, fecha):
    """
    """
    fig1, ax1 = mapa_base(llat, llon)
    tmin = extract_var(fecha, llat, llon, 'tmin')
    i_lat, i_lon, lat, lon = get_index_lat(fecha,llat,llon)
    # Datos para el Mapa
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(tmin) - 273.
    # Ploteamos el Mapa
    cMap = c.ListedColormap(['#A880C1', '#D1D5F0', '#ffffff'])
    bounds = np.array([-100., 0., 3., 100.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=3)
    CS = ax1.contourf(x, y, z, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    figure_name = 'tmin_' + fecha + '.png'
    plt.savefig(figure_name, dpi=200)

def mapa_pp(llat, llon, fecha):
    """
    """
    fig1, ax1 = mapa_base(llat, llon)
    ppacc = extract_var(fecha, llat, llon, 'pp')
    i_lat, i_lon, lat, lon = get_index_lat(fecha,llat,llon)
    # Datos para el Mapa
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(ppacc)
    # Ploteamos el Mapa
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#959392', '#5BC5F5',
                             '#E31903', '#7A0B0F'])
    bounds = np.array([0., 1., 20., 50., 100., 150., 1000.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=6)
    CS = ax1.contourf(x, y, z, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    figure_name = 'ppacum_' + fecha + '.png'
    plt.savefig(figure_name, dpi=200)

def extract_var(fecha, llat, llon, nomvar):
    " Devuelve las variables TMAX, TMIN, PPAcc"
    l_lat = llat
    l_lon = np.array(llon) % 360
    i_lat, i_lon, lat, lon = get_index_lat(fecha,llat,llon)

    # Comenzamos extraccion de variable
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p50/gfs' + fecha +\
         '/gfs_0p50_00z'
    file = netCDF4.Dataset(url)
    if nomvar == 'tmax':
        tmax2m = file.variables['tmax2m'][0:64,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        r_var = np.ma.max(tmax2m, axis=0)
        del(tmax2m)
    elif nomvar == 'tmin':
        tmin2m = file.variables['tmax2m'][0:64,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        r_var = np.ma.min(tmin2m, axis=0)
        del(tmin2m)
    elif nomvar == 'pp':
        # Precipitacion por 7 dias 2 --> 58 (datos cada 3 horas)
        ppinit = file.variables['apcpsfc'][2:58,i_lat[0]:i_lat[1]+1,
                                           i_lon[0]:i_lon[1]+1]
        r_var = np.ma.sum(ppinit, axis=0)
        del(ppinit)
    file.close()

    return r_var

def get_index_lat(fecha,llat,llon):
    """
    Get the index values for lat and lon to extract in requested square of data.
    """

    l_lat = llat
    l_lon = np.array(llon) % 360
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p50/gfs' + fecha +\
         '/gfs_0p50_00z'
    file = netCDF4.Dataset(url)
    flat  = file.variables['lat'][:]
    flon  = file.variables['lon'][:]

    lat = [a for a in flat if (a>=l_lat[0] and a<=l_lat[1])]
    i_lat = [np.where(flat==l_lat[0])[0][0], np.where(flat==l_lat[1])[0][0]]
    lon = [a for a in flon if (a>=l_lon[0] and a<=l_lon[1])]
    i_lon = [np.where(flon==l_lon[0])[0][0], np.where(flon==l_lon[1])[0][0]]

    file.close()
    return i_lat, i_lon, lat, lon


if __name__ == '__main__':
    mydate='20190307'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    mapa_pp(l_lat, l_lon, mydate)
