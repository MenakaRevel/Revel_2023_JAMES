from mpl_toolkits.basemap import Basemap
import re
import matplotlib.pyplot as plt

# Read the latitude and longitude data from the text file
# fname="/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_conus_06min_org.txt"
fname="/cluster/data6/menaka/HydroDA/dat/CGLS_alloc_conus_06min_DIR.txt"
with open(fname, 'r') as f:
    lines = f.readlines()

lats = []
lons = []
for line in lines[1:]:
    fields = filter(None, re.split(' ', line.strip()))
    lon, lat = float(fields[2]), float(fields[3])
    lats.append(lat)
    lons.append(lon)

# Create a new map
m = Basemap(projection='cyl', lat_0=0, lon_0=0,
            resolution='l', area_thresh=1000.0,
            llcrnrlon=-130, llcrnrlat=20,
            urcrnrlon=-60, urcrnrlat=55)

# Draw the coastlines and country borders
m.drawcoastlines()
m.drawcountries()

# Convert the latitude and longitude coordinates to x and y coordinates on the map
x, y = m(lons, lats)
# for i in range(len(lats)):
#     print('({:.2f}, {:.2f}) -> ({:.2f}, {:.2f})'.format(
#         lons[i], lats[i], x[i], y[i]))
# Plot the locations as red dots
m.plot(x, y, 'ro', markersize=5)

# Show the map
# plt.show()
# plt.savefig("map_VS_used_org.jpg")
plt.savefig("map_VS_used_DIR.jpg")