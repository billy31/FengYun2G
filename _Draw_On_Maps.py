#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-1 下午2:13
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : _Draw_On_Maps.py
# @Software: PyCharm

import os
import datetime
import glob
import gdal
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import geoplot
from Revision_FY2G import common_functions as cf
from Temp import Local_Processing_draw_maps as dm
from mpl_toolkits.basemap import Basemap
from Revision_FY2G import _HDF_Init as hdfx
from Temp import Final_Functions_modis_processing as modis

lonCenter = 104.5


def fengyun_draw_on_origin_data(date, hour, total_firepx, Xranges, Yranges, subplot, drawcb=None,
        geofiles = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G/NOM_ITG/NOM_ITG_2288_2288(0E0N)_LE.dat',
        FULLDISKdir='/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G/DATA/hdf/'):

    if date.__class__ == str:
        date_inthisalgorithm = date
    else:
        date_inthisalgorithm = datetime.datetime.strptime(date, '%Y%m%d')

    geog_FULLDISK = gdal.Open(geofiles)
    lon_FULLDISK = geog_FULLDISK.GetRasterBand(1).ReadAsArray() + lonCenter
    lat_FULLDISK = geog_FULLDISK.GetRasterBand(2).ReadAsArray()

    cmap = 'RdBu_r'
    try:
        data_FULLDISK_name = sorted(glob.glob(FULLDISKdir + '*' + date_inthisalgorithm + '_' +
                                              hour.__str__().zfill(2) + '00.hdf'))[0]
        data_FULLDISK = gdal.Open(data_FULLDISK_name)
        data_FULLDISK_subs_MIR = gdal.Open(data_FULLDISK.GetSubDatasets()[9][0])
        data_FULLDISK_subs_CAL = gdal.Open(data_FULLDISK.GetSubDatasets()[3][0])
        mir_FULLDISK_RAW = hdfx.cal_values(data_FULLDISK_subs_CAL.GetRasterBand(1).ReadAsArray(),
                                           data_FULLDISK_subs_MIR.GetRasterBand(1).ReadAsArray())

        if Xranges == range(13) and Yranges == range(13):
            lon_FULLDISK = np.ma.masked_greater_equal(lon_FULLDISK, 300 + lonCenter)
            lat_FULLDISK = np.ma.masked_greater_equal(lat_FULLDISK, 300)
            mir_FULLDISK_RAW = np.ma.masked_less_equal(mir_FULLDISK_RAW, 0)
            mymap = Basemap(projection='geos', lon_0=lonCenter, resolution='h', ax=subplot)
            mymap.drawcoastlines(linewidth=0.5)
            xs_Part, ys_FULLDISK = mymap(lon_FULLDISK, lat_FULLDISK)
            mymap.contourf(xs_Part, ys_FULLDISK, mir_FULLDISK_RAW, levels=np.linspace(200, 350, 300))
            if drawcb:
                colormesh = mymap.pcolormesh(xs_Part, ys_FULLDISK, mir_FULLDISK_RAW, cmap=cmap)
        else:
            WID = cf.WID
            LON = lon_FULLDISK[np.min(Xranges) * WID: (np.max(Xranges) + 1) * WID,
                  np.min(Yranges) * WID: (np.max(Yranges) + 1) * WID]
            LAT = lat_FULLDISK[np.min(Xranges) * WID: (np.max(Xranges) + 1) * WID,
                  np.min(Yranges) * WID: (np.max(Yranges) + 1) * WID]
            lllon, lllat, urlon, urlat = LON[-1, 0], LAT[-1, 0], LON[0, -1], LAT[0, -1]
            if lllon == 300 + lonCenter:
                lllon = np.min(np.ma.masked_greater_equal(LON, 300))
            if lllat == 300:
                lllat = np.min(np.ma.masked_greater_equal(LAT, 300))
            if urlon == 300 + lonCenter:
                urlon = np.max(np.ma.masked_greater_equal(LON, 300))
            if urlat == 300:
                urlat = np.max(np.ma.masked_greater_equal(LAT, 300))

            # LON = np.ma.masked_greater_equal(LON, 300)
            # LAT = np.ma.masked_greater_equal(LAT, 300)
            if np.mean(mir_FULLDISK_RAW[np.min(Xranges) * WID: (np.max(Xranges) + 1) * WID,
                           np.min(Yranges) * WID: (np.max(Yranges) + 1) * WID]) == 0:
                subplot.set_xticks([])
                subplot.set_yticks([])
                return 1
            else:
                mir_to_draw = mir_FULLDISK_RAW[np.min(Xranges) * WID: (np.max(Xranges) + 1) * WID,
                              np.min(Yranges) * WID: (np.max(Yranges) + 1) * WID]
                if np.min(mir_FULLDISK_RAW[np.min(Xranges) * WID: (np.max(Xranges) + 1) * WID,
                           np.min(Yranges) * WID: (np.max(Yranges) + 1) * WID]) == 0:
                    subplot.set_xticks([])
                    subplot.set_yticks([])
                    return 1
                    # invisiblemap = Basemap(projection='geos', lon_0=lonCenter, resolution=None)
                    # print('Unit x: %.3f | Unit y: %.3f' %
                    #       ((invisiblemap.urcrnrx/13), (invisiblemap.urcrnry/13)))
                    # print('Lx: %.3f - Ux: %.3f\nLy: %.3f - Uy: %.3f' %
                    #       ((invisiblemap.urcrnrx / 13 * np.min(Yranges)),
                    #        (invisiblemap.urcrnry / 13 * (np.max(Yranges) + 1)),
                    #        (invisiblemap.urcrnrx / 13 * np.min(Xranges)),
                    #        (invisiblemap.urcrnry / 13 * (np.max(Xranges) + 1))))
                    #
                    # mymap = Basemap(projection='geos',
                    #                 llcrnrx=invisiblemap.urcrnrx/13*np.min(Yranges),
                    #                 llcrnry=invisiblemap.urcrnry/13*np.min(Xranges),
                    #                 urcrnrx=invisiblemap.urcrnrx/13*(np.max(Yranges)+1),
                    #                 urcrnry=invisiblemap.urcrnry/13*(np.max(Xranges)+1),
                    #                 lon_0=lonCenter, resolution='h', ax=subplot)
                    # mymap.drawcoastlines(linewidth=0.5)
                else:
                    mymap = Basemap(projection='geos', llcrnrlon=lllon, llcrnrlat=lllat,
                                    urcrnrlon=urlon, urcrnrlat=urlat,
                                    lon_0=lonCenter, resolution='h', ax=subplot)
                    xs_Part, ys_Part = mymap(LON, LAT)
                    mymap.contourf(xs_Part, ys_Part, mir_to_draw, levels=np.linspace(200, 350, 300), cmap=cmap)
                    mymap.drawcoastlines(linewidth=0.5)
                    if drawcb:
                        colormesh = mymap.pcolormesh(xs_Part, ys_Part, mir_to_draw, cmap=cmap)
                        mymap.colorbar(colormesh)
    except:
        mymap = Basemap(ax=subplot)

    if total_firepx.__len__() > 0:
        # subplot.set_title('Total fire pixels: %3d', total_firepx.__len__())
        for i in iter(range(total_firepx.__len__())):
            lonpx, latpx = total_firepx[i]
            xpx, ypx = mymap(lonpx, latpx)
            mymap.scatter(xpx, ypx, marker='x', c='b')

    # plt.show()
    return mymap


# modisfirepd = cf.modis_monthly_data(monthlypath, date)


if __name__ == '__main__':

    indir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing/'  # '/home/lzy/'
    outdir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing_PFires/'  # '/home/lzy/figs/stable/'
    monthlypath = '/home/lzy/monthly/'
    hours = [10]
    date = '20160110'
    xranges = range(13)
    yranges = range(13)
    for hour in hours:
        cols, rows = 13, 13
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(18, 18))
        for x in xranges:
            for y in yranges:
                subs = x.__str__().zfill(2) + y.__str__().zfill(2)
                ax = axs[x, y]
                # ax.set_title('V: %2d | H: %2d' % (x, y))
                firepx = []
                # cf.drawbkmaps_single(indir, date, hour, x, y, ax)
                singleMAP = fengyun_draw_on_origin_data(date, hour, firepx, x, y, ax)
                print(x, y)
                # firepx = cf.fengyun_ts_algorithm(indir, outdir, date, hour, x, y, duration=20)
                # for i in range(modisfirepd.shape[0]):
                #     modfiretime = datetime.datetime.strptime(modisfirepd['YYYYMMDD'][i] + '_' +
                #                                              modisfirepd['HHMM'][i], '%Y%m%d_%H%M')
                #     if minlon <= float(modisfirepd['lon'][i]) <= maxlon and \
                #             minlat <= float(modisfirepd['lat'][i]) <= maxlat and \
                #             hour <= modfiretime.hour < hour + 1:
                #         lonpx, latpx = modisfirepd['lon'][i], modisfirepd['lat'][i]
                #         xpx, ypx = singleMAP(lonpx, latpx)
                #         singleMAP.scatter(xpx, ypx, marker='o', c='', edgecolor='r')
        #        fig.show()
        fig.savefig('/home/lzy/Fig 3. full disk new titled %s.png' % date, dpi=600)

    # indir = '/home6/FY2G/subsets/'
    # outdir = '/home6/FY2G/pfire/'
    # monthlypath = '/home6/monthly/'
    # geofiledir = '/home2/FY2G/NOM_ITG_2288_2288(0E0N)_LE/'
    # hdffiledir = '/home2/FY2G/'
    # dates = ['20160126']
    # hours = [1]
    # # xranges = range(13)
    # # yranges = range(13)
    # # fig, axs = plt.subplots(ncols=13, nrows=13, figsize=(26, 26))
    #
    # xranges = [7, 8, 9]
    # yranges = [2, 3, 4]
    # fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(8, 8))
    # for date in dates:
    #     modisfirepd = cf.modis_monthly_data(monthlypath, date)
    #     for hour in hours:
    #         for xid, x in enumerate(xranges):
    #             for yid, y in enumerate(yranges):
    #                 print(x, y)
    #                 ax = axs[xid, yid]
    #                 firepx = cf.fengyun_ts_algorithm(indir, outdir, date, hour, x, y)
    #                 singleMAP = fengyun_draw_on_origin_data(date, hour, [], [x], [y], ax)
    #     for i in range(modisfirepd.shape[0]):
    #         modfiretime = datetime.datetime.strptime(modisfirepd['YYYYMMDD'][i] + '_' +
    #                                                  modisfirepd['HHMM'][i], '%Y%m%d_%H%M')
    #         if 40.0 <= float(modisfirepd['lon'][i]) <= 180.0:
    #             lonpx, latpx = modisfirepd['lon'][i], modisfirepd['lat'][i]
    #             xpx, ypx = singleMAP(lonpx, latpx)
    #             singleMAP.scatter(xpx, ypx, marker='o', c='', edgecolor='r')
    #
    # fig.savefig('/home/lzy/china_area_%s.png' % date, dpi=600)
    # print('Finished')
    # for date in dates:
    #     fyallday = []
    #     modallday = []
    #     fig, ax = plt.subplots(ncols=6, nrows=4, figsize=(18, 12))
    #     modisfirepd = cf.modis_monthly_data(monthlypath, date)
    #     for ax_id, hour in enumerate(hours):
    #         axs = ax[int(ax_id/6), ax_id % 6]
    #         axs.set_title(date+' ' + hour.__str__().zfill(2)+':'+'00')
    #         # fengyunfire = cf.fengyun_ts_algorithm(indir, outdir, date, hour, xranges, yranges)
    #         # fyallday.append(fengyunfire)
    #         # fengyunfire = []
    #         singleMAP = fengyun_draw_on_origin_data(date, hour, [], xranges, yranges, axs)
    #                                                 # geofiles='/home/lzy/figs/NOM_ITG_2288_2288(0E0N)_LE.dat',
    #                                                 # FULLDISKdir='/home/lzy/figs/')
    #         # FULLMAP = fengyun_draw_on_origin_data(date, hour, fengyunfire, xranges, yranges)
    #         # #, geofiles=geofiledir, FULLDISKdir=hdffiledir)
    #
    #         for i in range(modisfirepd.shape[0]):
    #             modfiretime = datetime.datetime.strptime(modisfirepd['YYYYMMDD'][i]+'_' +
    #                                                      modisfirepd['HHMM'][i], '%Y%m%d_%H%M')
    #             if 40.0 <= float(modisfirepd['lon'][i]) <= 180.0 and \
    #                 hour <= modfiretime.hour < hour+1:
    #                 lonpx, latpx = modisfirepd['lon'][i], modisfirepd['lat'][i]
    #                 xpx, ypx = singleMAP(lonpx, latpx)
    #                 singleMAP.scatter(xpx, ypx, marker='o', c='', edgecolor='r')
    #
    #     fig.savefig('/home/lzy/aust_test_modisfire_new_1_%s.png' % date, dpi=600)
    #     plt.show()
    #     print(1)
    #     # doable
