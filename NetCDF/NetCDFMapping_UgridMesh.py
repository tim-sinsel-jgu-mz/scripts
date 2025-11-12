import netCDF4 as nc
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



# Filepath for the NetCDF file
ncfile = 'D:/develop/projects/NetCDFMappingTest/newugridmesh.nc'


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


time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'

levels = ncfile.createVariable('levels', np.float64, ('levels',))
levels.units = 'meters'
levels.long_name = 'levels'




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


# Define dimensions
nMesh2d_node = 10   # Number of 2D mesh nodes
nMesh2d_face = 8    # Number of 2D mesh faces
nMesh2d_edge = 15   # Number of 2D mesh edges


ncfile.createDimension("nMesh2d_node", nMesh2d_node)
ncfile.createDimension("nMesh2d_face", nMesh2d_face)
ncfile.createDimension("nMesh2d_edge", nMesh2d_edge)


# Define variables for mesh topology
mesh2d = ncfile.createVariable("mesh2d", "i4")
mesh2d.cf_role = "mesh_topology"
mesh2d.topology_dimension = 2
mesh2d.node_coordinates = "Mesh2d_node_x Mesh2d_node_y"
mesh2d.face_node_connectivity = "Mesh2d_face_nodes"

# Node coordinates
x = ncfile.createVariable("Mesh2d_node_x", "f4", ("nMesh2d_node",))
y = ncfile.createVariable("Mesh2d_node_y", "f4", ("nMesh2d_node",))

# Connectivity
face_nodes = ncfile.createVariable("Mesh2d_face_nodes", "i4", ("nMesh2d_face", "nMesh2d_edge"))
face_nodes.long_name = "Connectivity of face to nodes"

# 4D variable: Time, Levels, Face, Value
data_var = ncfile.createVariable("data", "f4", ("Time", "nVertLevels", "nMesh2d_face"), zlib=True)
data_var.units = "example_units"
data_var.long_name = "Example 4D data"

# Populate coordinates and connectivity with dummy data
x[:] = UTM_X[:]
y[:] = UTM_Y[:]
face_nodes[:, :] = np.random.randint(0, nMesh2d_node, size=(nMesh2d_face, nMesh2d_edge))


# create a 3D array of random numbers
data_arr = np.random.uniform(low=280,high=330,size=(ntimes,nlevs,nlats,nlons))

# Define a 3D variable to hold the data
temp = ncfile.createVariable('temp',np.float64,('time','levels',y_coords_string,x_coords_string)) # note: unlimited dimension is leftmost
temp.units = 'K' # degrees Kelvin
temp.standard_name = 'air_temperature' # this is a CF standard name
temp.grid_mapping = 'crs'
temp.coordinates = 'UTM_Y UTM_X'
# Write the data.  This writes the whole 3D netCDF variable all at once.
temp[:,:,:,:] = data_arr  # Appends data along unlimited dimension



# Global attributes
ncfile.title = "Example UGrid 3D Layered Mesh Topology"
ncfile.Conventions = "CF-1.8 UGrid-1.0"
ncfile.history = "Created for demonstration purposes"

# Close the file
ncfile.close()

print(f"NetCDF file {ncfile} created successfully!")
