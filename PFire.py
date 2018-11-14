#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-13 下午8:48
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : PFire.py
# @Software: PyCharm


import datetime
import glob
import os
import gdal
import numpy as np
from collections import Iterable as IT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# % matplotlib qt
# from Revision_FY2G import _Read_Init
# import _Read_Init
from common_functions import arr2TIFF
import pandas as pd

if __name__ == '__main__':
    temp_geo_dir = '/home2/FY2G/NOM_ITG_2288_2288(0E0N)_LE/'  # 'because of the error of geo files/temporally'
    geofile = temp_geo_dir + 'NOM_ITG_2288_2288(0E0N)_LE.dat'
    geoG = gdal.Open(geofile)
    geoshift = 104.5
    longitude = geoG.GetRasterBand(1).ReadAsArray()
    latitude = geoG.GetRasterBand(2).ReadAsArray()
    longitude+=geoshift

    stable_dir = '/home6/FY2G/subsets_stable/'
    out_dir = '/home6/FY2G/pfire/'
    origin_dir = '/home6/FY2G/subsets/'
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    # Multiple steps (SINGLE DAY VERSION)
    begindate = datetime.datetime(2015, 12, 20)
    flag = 1
    hour = 0
    pd_records = pd.DataFrame([])
    pd_file = '/home6/FY2G/pfire_list.csv'
    # while flag:
    for hour in iter(range(24)):
        date = begindate + datetime.timedelta(hours=hour)
        name_pre = 'FY2G_FDI_ALL_NOM_'
        name_sub1 = '_Delta_MOMS_MSTO.tif'
        print date
        for x in iter(range(13)):
            for y in iter(range(13)):
                subs = x.__str__().zfill(2) + y.__str__().zfill(2)
                os.chdir(stable_dir + subs + '/')
                filename = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + name_sub1
                if os.path.exists(filename) is False:
                    print 'No this file!'
                else:
                    xadded = x * 176
                    yadded = y * 176
                    g = gdal.Open(filename)
                    delta = g.GetRasterBand(1).ReadAsArray()
                    # LEN, WID = g.RasterYSize, g.RasterXSize
                    locs = np.where(delta > 0)
                    len_of_locs = locs[0].__len__()
                    if len_of_locs > 0:
                        print '%2d %d : %3d initial potential fire pixels' % (x, y, len_of_locs)
                        filename_origin = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '.tif'
                        origin_dirsub = origin_dir + subs + '/'
                        g_origin = gdal.Open(origin_dirsub + filename_origin)
                        tir = g_origin.GetRasterBand(1).ReadAsArray()
                        mir = g_origin.GetRasterBand(4).ReadAsArray()
                        count = 0
                        for i in iter(range(len_of_locs)):
                            _x, _y = locs[0][i], locs[1][i]
                            longitude_px, latitude_px = longitude[_x + xadded - 2, _y + yadded -2], \
                                                        latitude[_x + xadded - 2, _y + yadded -2]
                            d1 = delta[_x, _y], m1 = mir[_x, _y], t1 = tir[_x, _y]
                            d2 = m1 - t1
                            arr_MIR = mir[_x-2:_x+3, _y-2:_y+3]
                            arr_TIR = tir[_x-2:_x+3, _y-2:_y+3]
                            arr_Del = delta[_x-2:_x+3, _y-2:_y+3]
                            avg_mir_window = np.mean(arr_MIR)
                            avg_tir_window = np.mean(arr_TIR)
                            avg_del_window = np.mean(arr_TIR)
                            std_mir_window = np.std(arr_MIR)
                            std_tir_window = np.std(arr_TIR)
                            std_del_window = np.std(arr_Del)
                            pd_records = pd_records.append([[d1, m1, t1,
                                                             avg_mir_window, avg_tir_window, avg_del_window,
                                                             std_mir_window, std_tir_window, std_del_window,
                                                             longitude_px, latitude_px]])
                            # print "Delta: %3d | MIR: %3d  TIR: %3d" % (d1, m1, t1)
                    else:
                        print '%2d %d : No potential fires' % (x, y)

                    del g

        # hour += 1
    pd_records.to_csv(pd_file)
    print 'End'
