# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:51:54 2020

@author: odusi
"""

import gmplot
from PIL import Image
from PIL.ExifTags import TAGS
import os
import csv


class CameraConfig:
    """
    class contains the camera configuration for the DJI mavid dual enterprise
    """
    FOCAL_LENGTH = 2 ##mm
    SENSOR_WIDTH = 7.68
    def __init__(self, im):
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = im.size

        
    
    

class NoMetaData(Exception):
    pass



class GPSPlotter():
    def __init__(self, apikey,roi, height, filename):
        self.height = float(height)
        self.rois = list(map(self.get_roiInfo, roi))
        self.csv_writer(filename)
        print (self.rois)
        self.gmap = gmplot.GoogleMapPlotter(self.rois[0][0] + 0.01, self.rois[0][1] + 0.01,10, apikey=apikey)
        
    def plot_gps_points(self, filename):
        attractions = zip(*self.rois)
        attractions= list(attractions)
        lats = attractions[0]
        lons = attractions[1]
        
        self.gmap.heatmap(
                lats, lons,
                radius=40,
                weights=[1] * len(self.rois),
                gradient=[(0, 0, 255, 0), (0, 255, 0, 0.9), (255, 0, 0, 1)]
            )
        
        for roi in self.rois:
            self.gmap.marker(roi[0], roi[1], color='cornflowerblue')
            
            self.gmap.text(roi[0], roi[1], format(roi[2 ], ".2f")+ "m")
    
        self.gmap.draw(os.path.join("templates", filename))
    
    
    def get_roiInfo(self, seq):
        try :
            im = Image.open(seq.imgFilename)
            for (k,v) in im._getexif().items():
                if TAGS.get(k) == "GPSInfo":
                    lat, lon = self.convert_to_decimal_degrees(v[2], v[4]) 
                    gsd = self.get_gsd(CameraConfig(im))         
                    hotspot_size = seq.pixel_width * gsd
                    return lat, lon, hotspot_size          
                
        except AttributeError as e:
            raise NoMetaData("no GPS data available")
        
        
    def convert_to_decimal_degrees(self, lat_dms, lon_dms):
        lat = (lat_dms[0] + lat_dms[1]/60.0 + lat_dms[2]/3600.0) * -1 #fix for fact that gps parsing is wrong. we need to be below the equator
        
        lon = lon_dms[0] + lon_dms[1]/60.0 + lon_dms[2]/3600.0
        return lat, lon


    def get_gsd(self, camera):
        """
        GSD= (sensor_width * altitude * 100)/(focal_length * image_width) (m/pixel)
            
        """
        gsd = (camera.SENSOR_WIDTH * self.height) /(camera.FOCAL_LENGTH * camera.IMAGE_WIDTH)
        return gsd  
    
    def csv_writer(self, filename):
        with open(filename, "w", newline="") as csv_file:
            fieldnames = ["lat", "lon", "fire size(m)"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in self.rois:
                #pass
                writer.writerow({"lat":data[0], "lon":data[1], "fire size(m)":data[2]})

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