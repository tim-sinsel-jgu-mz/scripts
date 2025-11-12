import netCDF4 as nc
import numpy as np

# Filepath for the NetCDF file
output_file = "D:/develop/projects/NetCDFMappingTest/newugridmesh2.nc"

# Create the NetCDF file
ds = nc.Dataset(output_file, "w", format="NETCDF4")

# Define dimensions
nMesh2d_node = 10   # Number of 2D mesh nodes
nMesh2d_face = 8    # Number of 2D mesh faces
nMesh2d_edge = 15   # Number of 2D mesh edges
nVertLevels = 5     # Number of vertical levels
nTime = 10          # Time steps

ds.createDimension("nMesh2d_node", nMesh2d_node)
ds.createDimension("nMesh2d_face", nMesh2d_face)
ds.createDimension("nMesh2d_edge", nMesh2d_edge)
ds.createDimension("nVertLevels", nVertLevels)
ds.createDimension("Time", nTime)

# Define variables for mesh topology
mesh2d = ds.createVariable("mesh2d", "i4")
mesh2d.cf_role = "mesh_topology"
mesh2d.topology_dimension = 2
mesh2d.node_coordinates = "Mesh2d_node_x Mesh2d_node_y"
mesh2d.face_node_connectivity = "Mesh2d_face_nodes"

# Node coordinates in geographic coordinates (latitude, longitude)
lat = ds.createVariable("Mesh2d_node_lat", "f4", ("nMesh2d_node",))
lon = ds.createVariable("Mesh2d_node_lon", "f4", ("nMesh2d_node",))
lat.long_name = "latitude of 2D mesh nodes"
lon.long_name = "longitude of 2D mesh nodes"
lat.units = "degrees_north"
lon.units = "degrees_east"

# Node coordinates in UTM
utm_x = ds.createVariable("Mesh2d_node_utm_x", "f4", ("nMesh2d_node",))
utm_y = ds.createVariable("Mesh2d_node_utm_y", "f4", ("nMesh2d_node",))
utm_x.long_name = "UTM x-coordinate of 2D mesh nodes"
utm_y.long_name = "UTM y-coordinate of 2D mesh nodes"
utm_x.units = "meters"
utm_y.units = "meters"

# Connectivity
face_nodes = ds.createVariable("Mesh2d_face_nodes", "i4", ("nMesh2d_face", "nMesh2d_edge"))
face_nodes.long_name = "Connectivity of face to nodes"

# 4D variable: Time, Levels, Face, Value
data_var = ds.createVariable("data", "f4", ("Time", "nVertLevels", "nMesh2d_face"), zlib=True)
data_var.units = "example_units"
data_var.long_name = "Example 4D data"

# Populate lat/lon coordinates for a small area near Mainz, Germany
# Mainz: Approx. Latitude 49.992861, Longitude 8.247253
np.random.seed(42)  # For reproducibility
lat[:] = 49.992861 + np.random.uniform(-0.01, 0.01, nMesh2d_node)  # Random latitudes near Mainz
lon[:] = 8.247253 + np.random.uniform(-0.01, 0.01, nMesh2d_node)   # Random longitudes near Mainz

# Convert lat/lon to UTM using a simple approximate conversion
# Mainz falls in UTM zone 32N
k0 = 0.9996  # UTM scale factor
a = 6378137  # WGS84 major axis
f = 1 / 298.257223563  # WGS84 flattening
e2 = 2 * f - f ** 2  # Square of eccentricity
lon0 = 9  # Central meridian for UTM zone 32N

for i in range(nMesh2d_node):
    phi = np.radians(lat[i])
    lam = np.radians(lon[i])
    lam0 = np.radians(lon0)
    
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = e2 / (1 - e2) * np.cos(phi) ** 2
    A = np.cos(phi) * (lam - lam0)
    
    utm_x[i] = k0 * N * (A + (1 - T + C) * A ** 3 / 6)
    utm_y[i] = k0 * (N * np.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24))

# Populate connectivity and data with dummy values
face_nodes[:, :] = np.random.randint(0, nMesh2d_node, size=(nMesh2d_face, nMesh2d_edge))
data_var[:, :, :] = np.random.rand(nTime, nVertLevels, nMesh2d_face)

# Global attributes
ds.title = "Example UGrid 3D Layered Mesh with UTM Coordinates"
ds.Conventions = "CF-1.8 UGrid-1.0"
ds.history = "Created for demonstration purposes"
ds.geospatial_lat_min = lat[:].min()
ds.geospatial_lat_max = lat[:].max()
ds.geospatial_lon_min = lon[:].min()
ds.geospatial_lon_max = lon[:].max()
ds.utm_zone = "32N"

# Close the file
ds.close()

print(f"NetCDF file {output_file} created successfully!")
