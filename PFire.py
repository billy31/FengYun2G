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
from common_functions import image_preprocessing
from common_functions import trans
from common_functions import proj
# import pandas as pd
# image_preprocessing()

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
    begindate = datetime.datetime(2016, 1, 6)
    flag = 1
    hour = 0
    # pd_records = pd.DataFrame([])
    # pd_file = '/home6/FY2G/pfire_list_Aust.csv'
    # while flag:
    for hour in iter([5, 6, 7]):
        date = begindate + datetime.timedelta(hours=hour)
        name_pre = 'FY2G_FDI_ALL_NOM_'
        name_sub1 = '_Stable_MIR.tif'
        print date
        for x in iter([8, 9, 10]):
            for y in iter([9, 8, 7]):
                subs = x.__str__().zfill(2) + y.__str__().zfill(2)
                os.chdir(stable_dir + subs + '/')
                filename = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + name_sub1
                if os.path.exists(filename) is False:
                    print 'No this file!'
                else:
                    origin_dirsub = origin_dir + subs + '/'
                    filename_origin = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '.tif'
                    xadded = x * 176
                    yadded = y * 176
                    g1 = gdal.Open(filename)
                    g0 = gdal.Open(origin_dirsub + filename_origin)
                    stable = g1.GetRasterBand(1).ReadAsArray()
                    obsmir = g0.GetRasterBand(4).ReadAsArray()
                    obstir = g0.GetRasterBand(1).ReadAsArray()
                    ref = g0.GetRasterBand(5).ReadAsArray()
                    deltmt = obsmir - obstir
                    deltms = obsmir - stable
                    deltst = stable - obstir
                    deltMT = deltms - deltst

                    reqts1 = 1 * np.greater(deltms, 0)
                    reqts2_1 = 1 * np.less(deltst, np.mean(deltst))
                    reqts2_2 = 1 * np.greater_equal(deltst, 5)
                    reqts2 = 1 * np.equal(reqts2_1 + reqts2_2, 2)
                    reqts3 = 1 * np.greater(deltMT, 0)
                    reqts4 = 1 * np.less(ref, 20)

                    reqts = 1 * np.equal(reqts1 + reqts2 + reqts3 + reqts4, 4)
                    # plt.imsave('/home6/FY2G/pfire/reqts.png', reqts, dpi=600)

                    out_dir_pfire = out_dir + subs + '/'
                    if os.path.exists(out_dir_pfire) is False:
                        os.mkdir(out_dir_pfire)
                    os.chdir(out_dir_pfire)

                    outts1 = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_MO-MS>0.tif'
                    outts2 = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_MS-TO<AVG.tif'
                    outts3 = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_MOTO-2MS>0.tif'
                    outts4 = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_ref<20.tif'

                    arr2TIFF(reqts1, trans, proj, outts1, 1)
                    arr2TIFF(reqts2, trans, proj, outts2, 1)
                    arr2TIFF(reqts3, trans, proj, outts3, 1)
                    arr2TIFF(reqts4, trans, proj, outts4, 1)


                    len_of_locs = np.sum(reqts)
                    if len_of_locs > 0:
                        pfires = image_preprocessing(origin_dirsub + filename_origin)
                        reqct = 1 * np.equal(pfires, 1)
                        req = 1 * np.equal(reqts + reqct, 2)

                        outfile = name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_PFIRE.tif'
                        plt.imsave(out_dir + name_pre + date.strftime("%Y%m%d_%H%M") + '_' + subs + '_PFIRE.png',
                                   req, cmap='Set3', dpi=600)
                        arr2TIFF(req, trans, proj, outfile, 1)
                        print '[%2d %2d] Fire counts: %-4d ' % (x, y, np.sum(req))
                        print '---------------------------------'

                        # g_origin = gdal.Open(origin_dirsub + filename_origin)
                        # tir = g_origin.GetRasterBand(1).ReadAsArray()
                        # mir = g_origin.GetRasterBand(4).ReadAsArray()
                        # delta_value = mir - tir
                        # count = 0
                        # for i in iter(range(len_of_locs)):
                        #     _x, _y = locs[0][i], locs[1][i]
                        #     longitude_px, latitude_px = longitude[_x + xadded - 2, _y + yadded -2], \
                        #                                 latitude[_x + xadded - 2, _y + yadded -2]
                        #     d1 = delta_value[_x, _y]
                        #     m1 = mir[_x, _y]
                        #     t1 = tir[_x, _y]
                        #     d2 = m1 - t1
                        #     arr_MIR = mir[_x-2:_x+3, _y-2:_y+3]
                        #     arr_TIR = tir[_x-2:_x+3, _y-2:_y+3]
                        #     arr_Del = delta[_x-2:_x+3, _y-2:_y+3]
                        #     avg_mir_window = np.mean(arr_MIR)
                        #     avg_tir_window = np.mean(arr_TIR)
                        #     avg_del_window = np.mean(arr_TIR)
                        #     std_mir_window = np.std(arr_MIR)
                        #     std_tir_window = np.std(arr_TIR)
                        #     std_del_window = np.std(arr_Del)
                        #     if m1 > avg_mir_window and d2 > avg_del_window:
                        #         pd_records = pd_records.append([[d1, m1, t1, d2,
                        #                                          avg_mir_window, avg_tir_window, avg_del_window,
                        #                                          std_mir_window, std_tir_window, std_del_window,
                        #                                          longitude_px, latitude_px]])
                        #         count+=1
                        #     # print "Delta: %3d | MIR: %3d  TIR: %3d" % (d1, m1, t1)

                        # print '%2d %2d : %3d initial potential fire pixels' % (x, y, count)
                    else:
                        # print '%2d %d : No potential fires' % (x, y)
                        print '[%2d %2d] Fire counts: %-4d ' % (x, y, 0)
                        print '---------------------------------'
                    del g0, g1

        # hour += 1
    # pd_records.to_csv(pd_file)
    print 'End'
