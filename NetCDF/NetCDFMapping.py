from netCDF4 import Dataset 
import numpy as np
from pyproj import Transformer
from pyproj import CRS

import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    radAngle = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(radAngle) * (px - ox) - math.sin(radAngle) * (py - oy)
    qy = oy + math.sin(radAngle) * (px - ox) + math.cos(radAngle) * (py - oy)
    return qx, qy



ncfile = Dataset('D:/develop/projects/NetCDFMappingTest/new.nc',mode='w',format='NETCDF4') 
print(ncfile)

x_coords_string = 'rUTM_X'
y_coords_string = 'rUTM_Y'

lon_dim = ncfile.createDimension(x_coords_string, 30)    # longitude axis
lat_dim = ncfile.createDimension(y_coords_string, 20)     # latitude axis
lev_dim = ncfile.createDimension('levels', 15)     # latitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).

# Define two variables with the same names as dimensions,
# a conventional way to define "coordinate variables".

testcrs = CRS("EPSG:32632")
cf_grid_mapping = testcrs.to_cf()
wkt = testcrs.to_wkt()

crs = ncfile.createVariable('crs', np.int64, ())
'''
crs.epsg_code = '32632' 
crs.grid_mapping_name = "transverse_mercator"
crs.semi_major_axis = 6378137 
crs.semi_minor_axis = 6356752.314245 
crs.inverse_flattening = 298.257223563 
crs.latitude_of_projection_origin = 0.
crs.longitude_of_central_meridian = 9.0 
crs.false_easting = 500000.0 
crs.false_northing = 0. 
crs.scale_factor_at_central_meridian = 0.9996012717 

crs.PROJCRS = "WGS 84 / UTM zone 32N"
crs.BASEGEOGCRS = "WGS 84"
crs.proj4 = proj4str
crs.crs_wkt = "PROJCRS[WGS 84 / UTM zone 32N,BASEGEOGCRS[WGS 84,ENSEMBLE[World Geodetic System 1984 ensemble,MEMBER[World Geodetic System 1984 (Transit)],MEMBER[World Geodetic System 1984 (G730)],MEMBER[World Geodetic System 1984 (G873)],MEMBER[World Geodetic System 1984 (G1150)],MEMBER[World Geodetic System 1984 (G1674)],MEMBER[World Geodetic System 1984 (G1762)],MEMBER[World Geodetic System 1984 (G2139)],ELLIPSOID[WGS 84,6378137,298.257223563,  LENGTHUNIT[metre,1]],ENSEMBLEACCURACY[2.0]],PRIMEM[Greenwich,0,ANGLEUNIT[degree,0.0174532925199433]],ID[EPSG,4326]],CONVERSION[UTM zone 32N,METHOD[Transverse Mercator,ID[EPSG,9807]],PARAMETER[Latitude of natural origin,0,ANGLEUNIT[degree,0.0174532925199433],ID[EPSG,8801]],PARAMETER[Longitude of natural origin,9,ANGLEUNIT[degree,0.0174532925199433],ID[EPSG,8802]],PARAMETER[Scale factor at natural origin,0.9996,SCALEUNIT[unity,1],ID[EPSG,8805]],PARAMETER[False easting,500000,LENGTHUNIT[metre,1],ID[EPSG,8806]],PARAMETER[False northing,0,LENGTHUNIT[metre,1],ID[EPSG,8807]]],CS[Cartesian,2],AXIS[(E),east,ORDER[1],LENGTHUNIT[metre,1]],AXIS[(N),north,ORDER[2],LENGTHUNIT[metre,1]],USAGE[SCOPE[Navigation and medium accuracy spatial referencing.],AREA[Between 6°E and 12°E, northern hemisphere between equator and 84°N, onshore and offshore. Algeria. Austria. Cameroon. Denmark. Equatorial Guinea. France. Gabon. Germany. Italy. Libya. Liechtenstein. Monaco. Netherlands. Niger. Nigeria. Norway. Sao Tome and Principe. Svalbard. Sweden. Switzerland. Tunisia. Vatican City State.],BBOX[0,6,84,12]],ID[EPSG,32632]]"
'''
crs.angle_of_rotation_from_east_to_x = 30. 
crs.crs_wkt = wkt


rUTM_Y = ncfile.createVariable(y_coords_string, np.float32, (y_coords_string,))
rUTM_Y.units = 'meters'
rUTM_Y.standard_name = 'grid_projection_y_coordinate'
rUTM_Y.long_name = 'projected y coordinate in rotated transverse mercator grid'


rUTM_X = ncfile.createVariable(x_coords_string, np.float32, (x_coords_string,))
rUTM_X.units = 'meters'
rUTM_X.standard_name = 'grid_projection_x_coordinate'
rUTM_X.long_name = 'projected X coordinate in rotated transverse mercator grid'


UTM_Y = ncfile.createVariable('UTM_Y', np.float32, (y_coords_string,x_coords_string))
UTM_Y.units = 'meters'
UTM_Y.standard_name = 'projection_y_coordinate'
UTM_Y.long_name = 'projection_y_coordinate'


UTM_X = ncfile.createVariable('UTM_X', np.float32, (y_coords_string,x_coords_string))
UTM_X.units = 'meters'
UTM_X.standard_name = 'projection_x_coordinate'
UTM_X.long_name = 'projection_x_coordinate'



'''
gridsI = ncfile.createVariable('gridsI', np.float32, (x_coords_string,))
gridsI.units = 'meters'
gridsI.long_name = 'distance'
gridsJ = ncfile.createVariable('gridsJ', np.float32, (y_coords_string,))
gridsJ.units = 'meters'
gridsJ.long_name = 'distance'

lon = ncfile.createVariable('lon', np.float32, (x_coords_string,))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
lat = ncfile.createVariable('lat', np.float32, (y_coords_string,))
lat.units = 'degrees_north'
lat.long_name = 'latitude'

# Write latitudes, longitudes.
# Note: the ":" is necessary in these "write" statements
lon = np.zeros(nlons)
lat = np.zeros(nlats)
lon[:] = 8 + (0.1/nlons)*np.arange(nlons) 
lat[:] = 50. + (0.1/nlats)*np.arange(nlats) 

transformer = Transformer.from_crs("4326", "32632")
utm_y, utm_x = transformer.transform(lat, lon)

UTM_X[:] = utm_x
UTM_Y[:] = utm_y

gridsI[:] = (1/nlons)/2 + (1/nlons)*np.arange(nlons) 
gridsJ[:] = (1/nlats)/2 + (1/nlats)*np.arange(nlats) 

'''
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'

levels = ncfile.createVariable('levels', np.float64, ('levels',))
levels.units = 'meters'
levels.long_name = 'levels'

# Define a 3D variable to hold the data
temp = ncfile.createVariable('temp',np.float64,('time','levels',y_coords_string,x_coords_string)) # note: unlimited dimension is leftmost
temp.units = 'K' # degrees Kelvin
temp.standard_name = 'air_temperature' # this is a CF standard name
temp.grid_mapping = 'crs'
temp.coordinates = 'UTM_Y UTM_X'


nlats = len(lat_dim)
nlons = len(lon_dim)
nlevs = len(lev_dim)
ntimes = 3


rUTM_X[:] = 444444. + 3 * np.arange(nlons) 
rUTM_Y[:] = 5538888. + 3 * np.arange(nlats) 
levels[:] = 5 * np.arange(nlevs) 

UTM_X_2DAr =  np.zeros((nlats, nlons))
UTM_Y_2DAr =  np.zeros((nlats, nlons))

UTM_X_2DAr_rot =  np.zeros((nlats, nlons))
UTM_Y_2DAr_rot =  np.zeros((nlats, nlons))

for j in range(nlats):
    UTM_Y_2DAr[j,:] = rUTM_Y[j] 
for i in range(nlons):
    UTM_X_2DAr[:,i] = rUTM_X[i]     

origin = UTM_Y_2DAr[0,0], UTM_X_2DAr[0,0]
angle = 30 
for j in range(nlats):
    for i in range(nlons):
        UTM_X_2DAr_rot[j,i],UTM_Y_2DAr_rot[j,i] = rotate(origin, (UTM_Y_2DAr[j,i], UTM_X_2DAr[j,i]), angle)

UTM_Y[:,:] = UTM_Y_2DAr_rot[:,:] 
UTM_X[:,:] = UTM_X_2DAr_rot[:,:] 


# create a 3D array of random numbers
data_arr = np.random.uniform(low=280,high=330,size=(ntimes,nlevs,nlats,nlons))
# Write the data.  This writes the whole 3D netCDF variable all at once.
temp[:,:,:,:] = data_arr  # Appends data along unlimited dimension



ncfile.close()
print('Dataset is closed!')