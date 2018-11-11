#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-10 下午8:15
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : _HDF_Init.py
# @Software: PyCharm



import datetime
import glob
import os
import gdal
import numpy as np
import _Read_Init
import matplotlib.pyplot as plt


def cal_values(cal_index, data_array):
    out_data = np.zeros(data_array.shape, dtype=np.float32)
    for i, cal in enumerate(cal_index[0]):
        out_data[np.where(data_array == i)] = cal_index[0, i]
    return out_data


def __HDF_Init__(_in_dir='/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G/DATA/hdf/',
                 _out_dir='/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing/',
                 _geo_dir='/home/lzy/figs/NOM_ITG_2288_2288(0E0N)_LE.dat'):
    SUB = 13
    os.chdir(_in_dir)
    list_of_aims = sorted(glob.glob("FY2G*2016011*.hdf"))
    hdfgdal = gdal.Open(list_of_aims[0])
    band1fygdal = gdal.Open(hdfgdal.GetSubDatasets()[6][0])
    # band2fygdal = gdal.Open(hdfgdal.GetSubDatasets()[7][0])
    # band3fygdal = gdal.Open(hdfgdal.GetSubDatasets()[8][0])
    # band4fygdal = gdal.Open(hdfgdal.GetSubDatasets()[9][0])
    # band5fygdal = gdal.Open(hdfgdal.GetSubDatasets()[10][0])
    # band6fygdal = gdal.Open(hdfgdal.GetSubDatasets()[12][0])

    WID = band1fygdal.RasterXSize
    LEN = band1fygdal.RasterYSize
    BDS = 6
    l = LEN / SUB
    w = WID / SUB
    trans = band1fygdal.GetGeoTransform()
    proj = band1fygdal.GetProjection()

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
        try:
            hdfgdal = gdal.Open(files)
            cloud_mask = (hdfgdal.GetSubDatasets().__len__() == 18)
            # Read full disk into data
            for band in iter(range(BDS)):
                if band != 5:
                    calgdal = gdal.Open(hdfgdal.GetSubDatasets()[band][0])
                    fygdal = gdal.Open(hdfgdal.GetSubDatasets()[6+band][0])
                    full_disk[band] = cal_values(calgdal.GetRasterBand(1).ReadAsArray(),
                                                 fygdal.GetRasterBand(1).ReadAsArray())
                    del fygdal
                else:
                    if cloud_mask:
                        fygdal = gdal.Open(hdfgdal.GetSubDatasets()[12][0])
                        full_disk[band] = fygdal.GetRasterBand(1).ReadAsArray()
                        del fygdal

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
                            left = (_x+1)*w+2 if (_x+1)*w+2<WID else WID
                            top = _y*l-2 if _y*l-2>=0 else 0
                            bottom = (_y+1)*l+2 if (_y+1)*l+2<LEN else LEN

                            r1 = 2 if right==0 else 0
                            l1 = w+2 if left==WID else w+4
                            t1 = 2 if top==0 else 0
                            b1 = l+2 if bottom==LEN else l+4

                            subsets[band, r1:l1, t1:b1] = full_disk[band, right:left, top:bottom]

                            # Create related geo file once with buffer
                            if flag_or_geo_files:
                                _out_name_geo = _out_dir + subs + 'FY2G_ALL_' \
                                            + _x.__str__().zfill(2) + _y.__str__().zfill(2) + '_geo.dat'
                                subsets_geo = np.zeros((w + 4, l + 4))
                                subsets_geo[r1:l1, t1:b1] = geoarr[right:left, top:bottom]
                                _Read_Init.arr2TIFF(subsets_geo, trans, proj, _out_name_geo, 1)
                                flag_or_geo_files = False

                        # print r1,l1, t1,b1, right,left, top,bottom
                        print _x.__str__().zfill(2), '  -  ',  _y.__str__().zfill(2)
                        print 'Subsets: %4d - %4d   |   Full disk: %4d - %4d' % (r1, l1, right, left)
                        print 'Subsets: %4d - %4d   |   Full disk: %4d - %4d' % (t1, b1, top, bottom)

                        _Read_Init.arr2TIFF(subsets, trans, proj, _out_name, 6)
                        print 'File %3d Section %2d - %2d Created' % (i+1, _x, _y)
                    else:
                        print 'File %3d Section %2d - %2d Exists' % (i+1, _x, _y)

            print '%4d files left\n' % (list_of_aims.__len__() - (i+1))
        except Exception as e:
            print e.__str__()

if __name__ == '__main__':
    __HDF_Init__()
