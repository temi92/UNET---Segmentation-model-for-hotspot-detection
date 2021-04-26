# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:51:54 2020

@author: odusi
"""

import gmplot
from PIL import Image
from PIL.ExifTags import TAGS
import os
class GPSPlotter():
    def __init__(self, apikey,images):
        self.gps_positions = list(map(self.get_gps_position, images))
        print (self.gps_positions)
        self.gmap = gmplot.GoogleMapPlotter(self.gps_positions[0][0] + 0.01, self.gps_positions[0][1] + 0.01,16, apikey=apikey)

    def plot_gps_points(self, filename):
        attractions = zip(*self.gps_positions)
        self.gmap.heatmap(
                *attractions,
                radius=40,
                weights=[1] * len(self.gps_positions),
                gradient=[(0, 0, 255, 0), (0, 255, 0, 0.9), (255, 0, 0, 1)]
            )
        
        for position in self.gps_positions:
            self.gmap.marker(position[0], position[1], color='cornflowerblue')
        self.gmap.draw(os.path.join("templates", filename))
    
    
    def get_gps_position(self, image_file):
        for (k,v) in Image.open(image_file)._getexif().items():
            if TAGS.get(k) == "GPSInfo":
                lat, lon = self.convert_to_decimal_degrees(v[2], v[4]) 
                return lat, lon
        raise Exception("no GPS data available")
        
        
    def convert_to_decimal_degrees(self, lat_dms, lon_dms):
        lat = lat_dms[0] + lat_dms[1]/60.0 + lat_dms[2]/3600.0
        
        lon = lon_dms[0] + lon_dms[1]/60.0 + lon_dms[2]/3600.0
        return lat, lon
            
"""
# Create the map plotter:
apikey = "AIzaSyA3Je_LyveY2Ot6Z7UD25JLwihOKs1o46M"


points = [
    (37.769901, -122.498331),
    (37.768645, -122.475328),
    (37.771478, -122.468677),
    (37.769867, -122.466102),
    (37.767187, -122.467496),
    (37.770104, -122.470436)
]
g = GPSPlotter(apikey, (37.766956, -122.448481))
g.plot_gps_points(points, "map1.html")
"""