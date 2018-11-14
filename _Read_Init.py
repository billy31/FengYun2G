#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-1 下午8:13
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : _Read_Init.py
# @Software: PyCharm


import datetime
import glob
import os
import gdal
import numpy as np
from collections import Iterable as IT


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


def __Init__(_in_dir='/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G/NONGLT/',
          _out_dir='/home/lzy/figs/pis/',
          _geo_dir='/home/lzy/figs/NOM_ITG_2288_2288(0E0N)_LE.dat'):
    SUB = 13
    os.chdir(_in_dir)
    list_of_aims = sorted(glob.glob("FY2G*.tif"))
    fygdal = gdal.Open(list_of_aims[0])
    WID = fygdal.RasterXSize
    LEN = fygdal.RasterYSize
    BDS = fygdal.RasterCount
    l = LEN / SUB
    w = WID / SUB
    trans = fygdal.GetGeoTransform()
    proj = fygdal.GetProjection()

    geogdal = gdal.Open(_geo_dir)
    geoarr = geogdal.GetRasterBand(1).ReadAsArray()

    for _x in iter(range(SUB)):
        for _y in iter(range(SUB)):
            subs = _x.__str__().zfill(2) + _y.__str__().zfill(2) + '/'
            if os.path.exists(_out_dir + subs) is False:
                os.mkdir(_out_dir + subs)
    print 'Total files: %4d' % list_of_aims.__len__()
    for i, files in enumerate(list_of_aims):
        full_disk = np.zeros((BDS, LEN, WID))
        fygdal = gdal.Open(files)
        for band in iter(range(BDS)):
            full_disk[band] = fygdal.GetRasterBand(band+1).ReadAsArray()
        for _x in iter(range(SUB)):
            for _y in iter(range(SUB)):
                flag_or_geo_files = True
                subs = _x.__str__().zfill(2) + _y.__str__().zfill(2) + '/'
                _out_name = _out_dir + subs + files.split('.')[0] + '_' \
                            + _x.__str__().zfill(2) + _y.__str__().zfill(2) + '.tif'
                if os.path.exists(_out_name) is False:
                    subsets = np.zeros((BDS, w + 4, l + 4))

                    # Get the subsets with buffer
                    for band in iter(range(BDS)):
                        right = _x*w-2 if _x*w-2>=0 else 0
                        left = (_x+1)*w+1 if (_x+1)*w+1<WID else WID-1
                        top = _y*l-2 if _y*l-2>=0 else 0
                        bottom = (_y+1)*l+1 if (_y+1)*l+1<LEN else LEN-1

                        r1 = 2 if right==0 else 0
                        l1 = w+1 if left==WID-1 else w+3
                        t1 = 2 if top==0 else 0
                        b1 = l+1 if bottom==LEN-1 else l+3

                        subsets[band, r1:l1, t1:b1] = full_disk[band, right:left, top:bottom]

                        # Create related geo file once with buffer
                        if flag_or_geo_files:
                            _out_name_geo = _out_dir + subs + 'FY2G_ALL_' \
                                        + _x.__str__().zfill(2) + _y.__str__().zfill(2) + '_geo.dat'
                            subsets_geo = np.zeros((w + 4, l + 4))
                            subsets_geo[r1:l1, t1:b1] = geoarr[right:left, top:bottom]
                            _Read_Init.arr2TIFF(subsets_geo, trans, proj, _out_name_geo, 1)
                            flag_or_geo_files = False

                    arr2TIFF(subsets, trans, proj, _out_name, 6)
                    print 'File %3d Section %2d - %2d Created' % (i+1, _x, _y)
                else:
                    print 'File %3d Section %2d - %2d Exists' % (i+1, _x, _y)

        print '%4d files left\n' % (list_of_aims.__len__() - (i+1))