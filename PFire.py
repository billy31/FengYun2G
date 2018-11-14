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
import _Read_Init

if __name__ == '__main__':
    stable_dir = '/home6/FY2G/subsets_stable/'
    out_dir = '/home6/FY2G/pfire/'
    origin_dir = '/home6/FY2G/subsets/'
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    # Multiple steps (SINGLE DAY VERSION)
    begindate = datetime.datetime(2015, 12, 20)
    flag = 1
    hour = 0
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

                        for i in iter(range(len_of_locs)):
                            _x, _y = locs[0][i], locs[1][i]
                            d1 = delta[_x, _y], m1 = mir[_x, _y], t1 = tir[_x, _y]
                            # print "Delta: %3d | MIR: %3d  TIR: %3d" % (d1, m1, t1)
                            d2 = m1 - t1
                            avg_window = np.mean(mir[_x-2:_x+3, _y-2:_y+3])


                    else:
                        print '%2d %d : No potential fires' % (x, y)

                    del g

        # hour += 1
    print 'End'
