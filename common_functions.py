#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午8:56
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : common_functions.py
# @Software: PyCharm

import gdal


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
                print 'north:%.3f\tsouth:%.3f\t|\twest:%.3f\teast:%.3f' % (latitude[x1, y1], latitude[x2, y2],
                                                                           longitude[x1, y1], longitude[x2, y2])


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

if __name__ == '__main__':
    where_are_the_locations()