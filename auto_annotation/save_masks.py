# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:34:10 2020

@author: odusi
"""

import os
import os.path as osp
import sys
import glob
from pathlib import Path
from mask_creation import Roi
import cv2
import argparse
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    args = parser.parse_args()
    
    if osp.exists(args.output_dir):
        
        print("output directory already exists:" , args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    
    for img_path in list(Path( os.getcwd(), args.input_dir).glob("*.jpg")):
        
        img_path = str(img_path)            
        im = cv2.imread(img_path)
        roi = Roi(im)
        image = roi.get_roi()
        #get file name and extension
        label_filename = osp.splitext(img_path)[0].split("\\")[-1]
        label_ext = osp.splitext(img_path)[1]
         
        path_name = osp.join(args.output_dir, label_filename+ label_ext)
        cv2.imwrite(path_name,  image)
        """
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
main()