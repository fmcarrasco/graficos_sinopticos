# Funciones para graficar mapas sinopticos
# para el pronóstico del informe ORA

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


def mapa_base(llat, llon):
    """
    Mapa base para graficar las variables
    """

    l_lat = llat
    l_lon = np.array(llon) % 360  #Pasamos lon en [-180, 180] a [0, 360]
    # Trabajamos con el SHAPEFILE de IGN para provincias
    fname = './shapefile/Provincias'
    shape_feature = cpf.ShapelyFeature(Reader(fname).geometries(),
                                   ccrs.PlateCarree(), edgecolor='black',
                                   facecolor='None', linewidth=0.5)
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
    #ax.add_feature(shape_feature)
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
            fontweight='bold', fontsize=14,
            transform=ccrs.Geodetic())
    return fig, ax

def legend_temp(ax, x, y, clr, txt, proj):
    """
    Define legends as square for specific values.
    x, y, clr  and txt MUST be arrays of same len to work.
    the proj value, comes from Cartopy projection used in ax axis.
    """
    for i in np.arange(0, len(x)):
        ax.add_patch(mpatches.Rectangle(xy=[x[i], y[i]], width=1.8, height=1.5,
                                         facecolor=clr[i],
                                         transform=proj))
        ax.text(x[i] - -2.2, y[i] - -0.2, txt[i], horizontalalignment='left',
                 fontweight='bold', fontsize=14, transform=ccrs.Geodetic())


def mapa_tmax(file, llat, llon, fecha, figure_name):
    """
    Mapa de temperaturas maximas durante la semana:
    Tmax>40 y 35<Tmax<=40
    """
    tmax = extract_var(fecha, file, llat, llon, 'tmax')
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    fig1, ax1 = mapa_base(llat, llon)
    # Datos para el Mapa
    # Pasamos lon en [0, 360] a [-180, 180]
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(tmax) - 273.
    z[lndsfc==0.] = np.nan
    # Ploteamos el Mapa
    cMap = c.ListedColormap(['#ffffff', '#FF6002', '#a70101'])
    bounds = np.array([-10., 35., 40., 100.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=3)
    CS = ax1.contourf(x, y, z, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    # Agregamos los datos para la leyenda
    legend_temp(ax1, [-60, -60], [-45, -47], ['#a70101', '#FF6002'],
                ['>40' + u'\u2103', '35 - 40 ' + u'\u2103'],
                ccrs.PlateCarree())
    plt.savefig(figure_name, dpi=200)

def mapa_tmin(file, llat, llon, fecha, figure_name):
    """
    Mapa de temperaturas minimas durante la semana:
    Tmin<0 y 0<=Tmin<3
    """
    tmin = extract_var(fecha, file, llat, llon, 'tmin')
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    fig1, ax1 = mapa_base(llat, llon)
    # Datos para el Mapa
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(tmin) - 273.
    # Solo puntos en superficie continental
    z[lndsfc==0.] = np.nan
    # Preparamos los intervalos y colores a usar
    cMap = c.ListedColormap(['#A880C1', '#D1D5F0', '#ffffff'])
    bounds = np.array([-100., 0., 3., 100.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=3)
    # Ploteamos la variable
    CS = ax1.contourf(x, y, z, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    # Agregamos los datos para la leyenda
    legend_temp(ax1, [-60, -60], [-45, -47], ['#D1D5F0', '#A880C1'],
                ['0 - 3 ' + u'\u2103', '< 0' + u'\u2103'],
                ccrs.PlateCarree())
    plt.savefig(figure_name, dpi=200)

def mapa_pp(file, llat, llon, fecha, figure_name):
    """
    Acumulado de precipitacion durante la semana
    """
    ppacc = extract_var(fecha, file, llat, llon, 'pp')
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    fig1, ax1 = mapa_base(llat, llon)
    # Datos para el Mapa
    # Pasamos lon en [0, 360] a [-180, 180]
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(ppacc)
    # Suavizamos el campo de PP (estetica ante todo)
    z1 = ndimage.gaussian_filter(z, sigma=1., order=0)
    z1[lndsfc==0.] = np.nan

    # Ploteamos el Mapa
    cMap = c.ListedColormap(['#ffffff', '#fffaaa', '#959392', '#5BC5F5',
                             '#E31903', '#7A0B0F'])
    bounds = np.array([0., 1., 20., 50., 100., 150., 1000.])
    norm = c.BoundaryNorm(boundaries=bounds, ncolors=6)
    CS = ax1.contourf(x, y, z1, levels=bounds, cmap=cMap, norm=norm,
                      transform=ccrs.PlateCarree())
    cbaxes = inset_axes(ax1, width="50%", height="2%", loc=4)
    cb = plt.colorbar(CS, cax=cbaxes, orientation='horizontal', drawedges=True)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.set_xticklabels(['0', '1', '20', '50', '100', '150', ''],
                          weight='bold', fontsize=13)
    plt.savefig(figure_name, dpi=200)


def mapa_temp(file, llat, llon, fecha, figure_name):
    """
    Mapa de temperaturas extremas durante la semana:
    Tmin<0 y 0<=Tmin<3
    Tmax>40 y 35<Tmin<=40
    """
    # Set URL, open the file and extract variables needed.
    tmax = extract_var(fecha, file, llat, llon, 'tmax')
    tmin = extract_var(fecha, file, llat, llon, 'tmin')
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # Datos para el Mapa
    fig1, ax1 = mapa_base(llat, llon)
    # de 0-360 a -180 - 180 en Longitud
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z1 = np.ma.getdata(tmin) - 273.
    z1[lndsfc==0.] = np.nan
    z2 = np.ma.getdata(tmax) - 273.
    z2[lndsfc==0.] = np.nan
    ########
    # Ploteamos el Mapa de Tmin
    # Consideramos un "Blanco" con alpha = 0 para que sea transparente (TUPLA)
    cMap1 = c.ListedColormap(['#A880C1', '#D1D5F0', (1,1,1,0.)])
    bounds1 = np.array([-100., 0., 3., 100.])
    norm1 = c.BoundaryNorm(boundaries=bounds1, ncolors=3)
    CS1 = ax1.contourf(x, y, z1, levels=bounds1, cmap=cMap1, norm=norm1,
                       transform=ccrs.PlateCarree())
    # Ploteamos el Mapa de Tmax
    # Consideramos un "Blanco" con alpha = 0 para que sea transparente (TUPLA)
    cMap2 = c.ListedColormap([(1,1,1,0.), '#FF6002', '#a70101'])
    bounds2 = np.array([-10., 35., 40., 100.])
    norm2 = c.BoundaryNorm(boundaries=bounds2, ncolors=3)
    CS2 = ax1.contourf(x, y, z2, levels=bounds2, cmap=cMap2, norm=norm2,
                       transform=ccrs.PlateCarree())
    # Leyenda TMAX
    legend_temp(ax1, [-60, -60], [-41, -43], ['#a70101', '#FF6002'],
                ['>40' + u'\u2103', '35 - 40 ' + u'\u2103'],
                ccrs.PlateCarree())
    # Leyenda TMIN
    legend_temp(ax1, [-60, -60], [-46, -48], ['#D1D5F0', '#A880C1'],
                ['0 - 3 ' + u'\u2103', '< 0' + u'\u2103'],
                ccrs.PlateCarree())
    plt.savefig(figure_name, dpi=200)


def mapa_landsfc(file, llat, llon, fecha, figure_name):
    """
    Mapa de puntos clasificados como AGUA (=0) o
    TIERRA (=1)
    """
    # Set URL, open the file and extract variables needed.
    lndsfc = extract_var(fecha, file, llat, llon, 'lndsfc')
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    # Datos para el Mapa
    fig1, ax1 = mapa_base(llat, llon)
    x = ((np.squeeze(np.asarray(lon)) - 180) % 360) - 180
    y = np.squeeze(np.asarray(lat))
    z = np.ma.getdata(lndsfc)
    CS = ax1.pcolormesh(x, y, z, transform=ccrs.PlateCarree())
    plt.savefig(figure_name, dpi=200)


def extract_var(fecha, file, llat, llon, nomvar):
    " Devuelve las variables TMAX, TMIN, PPAcc"
    l_lat = llat
    l_lon = np.array(llon) % 360
    i_lat, i_lon, lat, lon = get_index_lat(fecha, file, llat, llon)
    if nomvar == 'tmax':
        tmax2m = file.variables['tmax2m'][0:64,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        r_var = np.ma.max(tmax2m, axis=0)
        del(tmax2m)
    elif nomvar == 'tmin':
        tmin2m = file.variables['tmin2m'][0:64,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]
        r_var = np.ma.min(tmin2m, axis=0)
        del(tmin2m)
    elif nomvar == 'pp':
        # Precipitacion por 7 dias 2 --> 58 (datos cada 3 horas)
        ppinit = file.variables['apcpsfc'][2:58,i_lat[0]:i_lat[1]+1,
                                           i_lon[0]:i_lon[1]+1]
        r_var = np.ma.sum(ppinit, axis=0)
        del(ppinit)
    elif nomvar == 'lndsfc':
        r_var = file.variables['landsfc'][0,i_lat[0]:i_lat[1]+1,
                                          i_lon[0]:i_lon[1]+1]

    return r_var

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


if __name__ == '__main__':
    mydate='20190530'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    ofolder = 'e:/python/graficos_sinopticos/'
    #########################################################################
    # Graficos en 0.5 grados
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p50/gfs' + mydate +\
         '/gfs_0p50_00z'
    file = netCDF4.Dataset(url)
    nomfig = ofolder + 'PPACUM_05deg_' + mydate + '.png'
    mapa_pp(file, l_lat, l_lon, mydate, nomfig)
    nomfig = ofolder + 'TEX_05deg_' + mydate + '.png'
    mapa_temp(file, l_lat, l_lon, mydate, nomfig)
    file.close()
    #########################################################################
    # Graficos en 0.25 grados
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs' + mydate +\
         '/gfs_0p25_00z'
    file = netCDF4.Dataset(url)
    nomfig = ofolder + 'PPACUM_025deg_' + mydate + '.png'
    mapa_pp(file, l_lat, l_lon, mydate, nomfig)
    nomfig = ofolder + 'TEX_025deg_' + mydate + '.png'
    mapa_temp(file, l_lat, l_lon, mydate, nomfig)
    file.close()
