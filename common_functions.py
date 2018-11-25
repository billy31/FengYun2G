#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午8:56
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : common_functions.py
# @Software: PyCharm

import gdal
import os
import glob
import numpy as np


trans = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
proj = 'PROJCS["New_Projected_Coordinate_System",' \
       'GEOGCS["GCS_New_Geographic_Coordinate_System",' \
       'DATUM["WGS_1984",SPHEROID["WGS_84",6378137.0,298.257223563]],' \
       'PRIMEM["<custom>",104.5],UNIT["<custom>",0.048952]],' \
       'PROJECTION["Lambert_Conformal_Conic_2SP"],' \
       'PARAMETER["False_Easting",104.5],' \
       'PARAMETER["False_Northing",0.0],' \
       'PARAMETER["Central_Meridian",104.5],' \
       'PARAMETER["Standard_Parallel_1",60.0],' \
       'PARAMETER["Standard_Parallel_2",60.0],' \
       'PARAMETER["Scale_Factor",1.0],' \
       'PARAMETER["Latitude_Of_Origin",0.0],' \
       'UNIT["<custom>",5000.0]]'



def where_are_the_locations():
    temp_geo_dir = '/home2/FY2G/NOM_ITG_2288_2288(0E0N)_LE/'  # 'because of the error of geo files/temporally'
    geofile = temp_geo_dir + 'NOM_ITG_2288_2288(0E0N)_LE.dat'
    geoG = gdal.Open(geofile)
    geoshift = 104.5
    longitude = geoG.GetRasterBand(1).ReadAsArray()
    latitude = geoG.GetRasterBand(2).ReadAsArray()
    longitude+=geoshift
    for x in iter(range(13)):
        for y in iter(range(13)):
            x1 = x * 176
            y1 = y * 176
            x2 = (x+1) * 176 - 1
            y2 = (y+1) * 176 - 1

            if longitude[x1, y1] >= 300 or longitude[x2, y2] >= 300:
                continue
            else:
                print '%2d %2d\t| north:%.3f\tsouth:%.3f\t| west:%.3f\teast:%.3f' % \
                      (x, y, latitude[x1, y1], latitude[x2, y2], longitude[x1, y1], longitude[x2, y2])


def arr2TIFF(im_data, im_geotrans, im_proj, im_file, im_bands):
    datatype = gdal.GDT_Float32
    im_height, im_width = im_data[0].shape if im_bands>1 else im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(im_file, im_width, im_height, im_bands, datatype)  # 创建文件
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        if im_bands==1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def starrays(data, min, max):
    req_1 = 1 * np.greater_equal(data, min)
    req_2 = 1 * np.less_equal(data, max)
    req = 1 * np.equal(req_1+req_2, 2)
    return req


def image_preprocessing(inputfile):
    g = gdal.Open(inputfile)
    MIR_original = g.GetRasterBand(4).ReadAsArray()
    TIR_original = g.GetRasterBand(1).ReadAsArray()
    DELTA_original = MIR_original - TIR_original

    MIR = MIR_original[2:-2, 2:-2]
    TIR = TIR_original[2:-2, 2:-2]

    delta = MIR - TIR

    req_MIR = starrays(MIR, 290, 350)

    req_TIR = starrays(TIR, 270, 350)

    req_delta = 1 * np.greater_equal(delta, 10)
    reqs_basic = 1 * np.equal(req_MIR + req_TIR + req_delta, 3)

    req_MIR_new = 1 * np.greater_equal(MIR, np.mean(MIR[np.where(req_MIR)]))
    req_TIR_new = 1 * np.greater_equal(TIR, np.mean(TIR[np.where(req_TIR)]))
    print 'MIR AVG %-3.3f/ TIR AVG %-3.3f' % (np.mean(MIR[np.where(req_MIR)]), np.mean(TIR[np.where(req_TIR)]))
    req_IR = 1 * np.equal(req_MIR_new + req_TIR_new, 2)
    reqs_dynamic = 1 * np.equal(req_IR + reqs_basic, 2)
    locs = np.where(reqs_dynamic == True)
    length = np.sum(reqs_dynamic)
    outfile = np.zeros((180, 180))
    if length > 0:
        for i in iter(range(length)):
            x, y = locs[0][i] + 2, locs[1][i] + 2
            mir_array = MIR_original[x - 2:x + 3, y - 2:y + 3]
            tir_array = TIR_original[x - 2:x + 3, y - 2:y + 3]
            delta_array = mir_array - tir_array
            validpx = starrays(mir_array, 290, 350) == 1
            mir_avg = np.mean(mir_array[validpx])
            mir_std = np.std(mir_array[validpx])
            delta_avg = np.mean(delta_array[validpx])
            delta_std = np.std(delta_array[validpx])
            if MIR_original[x, y] > mir_avg + 0.5 * mir_std and \
                            DELTA_original[x, y] > delta_avg + 0.5 * delta_std:
                outfile[x, y] = 1

    return outfile



# if __name__ == '__main__':
#     where_are_the_locations()