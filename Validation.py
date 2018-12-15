#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-13 下午4:24
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : Validation.py
# @Software: PyCharm

import os
import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from Revision_FY2G import common_functions as cf
import pandas as pd


if __name__ == '__main__':
    monthlydir = '/home6/monthly/'
    indir = '/home6/FY2G/subsets/'
    outdir = '/home6/FY2G/validation/'
    if os.path.exists(outdir):
        os.mkdir(outdir)

    dates = np.arange(start=datetime.datetime(2016, 1, 1), stop=datetime.datetime(2016, 4, 1),
                      step=datetime.timedelta(days=1))
    # dates = ['20160107']
    xranges = range(7, 11)
    yranges = range(7, 11)

    getranges = cf.draw_ranges(xranges, yranges,
                               geofile='/home2/FY2G/NOM_ITG_2288_2288(0E0N)_LE/NOM_ITG_2288_2288(0E0N)_LE.dat')

    hours = range(24)
    for date_fmt in dates:
        modisallday = []; fengyunallday = []
        if date_fmt.__class__ == str:
            date = datetime.datetime.strptime(date_fmt, '%Y%m%d')
        elif date_fmt.__class__ == datetime.datetime:
            date = date_fmt
        else:
            date = date_fmt.astype(datetime.datetime)
        modispd = cf.modis_monthly_data(monthlydir, date)

        lon_req = np.logical_and(getranges[0] <= pd.to_numeric(modispd['lon']),
                                 getranges[2] >= pd.to_numeric(modispd['lon']))
        lat_req = np.logical_and(getranges[1] <= pd.to_numeric(modispd['lat']),
                                 getranges[3] >= pd.to_numeric(modispd['lat']))

        daily_modfire_count = modispd.loc[np.logical_and(lon_req, lat_req)].size/11
        modfiregeo = list(modispd.loc[np.logical_and(lon_req, lat_req)][['lon', 'lat']]._values)
        modfiregeo_f = [[float(i[0]), float(i[1])] for i in modfiregeo]
        modvalid = np.zeros((int(daily_modfire_count), 1))
        for hour in hours:
            for x in xranges:
                for y in yranges:
                    if [x, y] in cf.workable:
                        fengyunfire = cf.fengyun_ts_algorithm(indir, outdir, date, hour, x, y, 'Todatabase')
                        if fengyunfire != []:
                            [fengyunallday.append(i) for i in fengyunfire]
        if fengyunallday.__len__() > 0:
            fyvalid = np.zeros((fengyunallday.__len__(), 1))
            for fyID, fyfire in enumerate(fengyunallday):
                # _geoslib.Polygon
                # resolution = cf.calculate_resolution(fyfire)
                resolutions = [0.05, 0.05]
                fypoly = cf.px_bounding(fyfire, resolutions)
                modvalid, fyvalid[fyID] = cf.validation(fypoly, modfiregeo_f, modvalid)
                # print(fyID.__str__().zfill(3))

        print(60*'=' + '\n')
        print('%15s | %-15s | %-15s | %-15s' % (' ', 'MODIS True', 'FY True', 'Total'))
        print(15 * '-' + '---' + 15 * '-' + '---' + 15 * '-' + '---' + 15 * '-')
        print('%15s | %-15d | %-15s | %-15d' % ('MODIS Detect', np.sum(modvalid), '-', int(daily_modfire_count)))
        print('%15s | %-15s | %-15d | %-15d' % ('FY Detect', '-', np.sum(fyvalid), fengyunallday.__len__()))
        print(15 * '-' + '---' + 15 * '-' + '---' + 15 * '-' + '---' + 15 * '-')
        print('%15s | %-15.3f | %-15.3f | %-15s' % (
            ' ', np.sum(modvalid) / int(daily_modfire_count), np.sum(fyvalid) / fengyunallday.__len__(), ' '))

        # print(date)