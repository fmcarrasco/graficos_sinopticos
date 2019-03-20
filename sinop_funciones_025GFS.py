import numpy as np
import netCDF4
import scipy.ndimage as ndimage
import datetime as dt

def get_index_lat(fecha,llat,llon):
    """
    Get the index values for lat and lon to extract in requested square of data.
    """

    l_lat = llat
    l_lon = np.array(llon) % 360  # pasamos de [-180, 180] a [0, 360]
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs' + fecha +\
         '/gfs_0p25_00z'
    file = netCDF4.Dataset(url)
    flat  = file.variables['lat'][:]
    flon  = file.variables['lon'][:]

    lat = [a for a in flat if (a>=l_lat[0] and a<=l_lat[1])]
    i_lat = [np.where(flat==l_lat[0])[0][0], np.where(flat==l_lat[1])[0][0]]
    lon = [a for a in flon if (a>=l_lon[0] and a<=l_lon[1])]
    i_lon = [np.where(flon==l_lon[0])[0][0], np.where(flon==l_lon[1])[0][0]]

    file.close()
    return i_lat, i_lon, lat, lon

def get_index_time(fecha):
    """
    """
    url ='https://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs' + fecha +\
         '/gfs_0p25_00z'
    file = netCDF4.Dataset(url)
    aux = file.variables['time'][:]
    a = file.variables['time'].getncattr('units')
    c = netCDF4.num2date(aux, units=a)
    tiempo = [dt.datetime.strptime(str(tp), '%Y-%m-%d %H:%M:%S') for tp in c]
    file.close()
    return tiempo

if __name__ == '__main__':

    mydate = '20190308'
    l_lat = [-60., -20.]
    l_lon = [-80., -50.]
    i_lat, i_lon, lat, lon = get_index_lat(mydate,l_lat,l_lon)
    fechas = get_index_time(mydate)
    print(fechas[0])
