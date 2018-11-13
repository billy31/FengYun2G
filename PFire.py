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
        for x in iter([7,8,9]):
            for y in iter([7,8,9]):
                print x, y
                subs = x.__str__().zfill(2) + y.__str__().zfill(2)
                os.chdir(stable_dir + subs + '/')
                origin_dirsub = origin_dir + subs + '/'
                filename = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + name_sub1
                if os.path.exists(filename) is False:
                    print 'No this file!'
                else:
                    g = gdal.Open(filename)
                    delta = g.GetRasterBand(1).ReadAsArray()
                    LEN, WID = delta.RasterYSize, delta.RasterXSize
                    locs = np.where(delta)
                    if locs[0].__len__() > 0:
                        print x, y, ':', locs
                    else:
                        print x, y, ': No potential fires'


        # hour += 1
    print 'End'
