#!/bin/bash

mkdir -p train/pixel/
mkdir -p train/cdl_label/
mkdir -p train/road_label/
mkdir -p train/building_label/

# Wipe everything clean (in case we're changing the tile size, for example)
rm train/pixel/*tif
rm train/cdl_label/*tif
rm train/road_label/*tif
rm train/building_label/*tif

gdal_retile.py -ps 512 512 -targetDir train/pixel/ naip/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/pixel/ naip/m_4009001_sw_15_1_20170725.tif
gdal_retile.py -ps 512 512 -targetDir train/pixel/ naip/m_4109120_nw_15_1_20170819.tif

gdal_retile.py -ps 512 512 -targetDir train/cdl_label/ cdl_annotations/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/cdl_label/ cdl_annotations/m_4009001_sw_15_1_20170725.tif
gdal_retile.py -ps 512 512 -targetDir train/cdl_label/ cdl_annotations/m_4109120_nw_15_1_20170819.tif

gdal_retile.py -ps 512 512 -targetDir train/road_label/ road_annotations/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/road_label/ road_annotations/m_4009001_sw_15_1_20170725.tif
gdal_retile.py -ps 512 512 -targetDir train/road_label/ road_annotations/m_4109120_nw_15_1_20170819.tif

gdal_retile.py -ps 512 512 -targetDir train/building_label/ building_annotations/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/building_label/ building_annotations/m_4009001_sw_15_1_20170725.tif
gdal_retile.py -ps 512 512 -targetDir train/building_label/ building_annotations/m_4109120_nw_15_1_20170819.tif
