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
import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt

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


def check_exist(aim, db, sub, duration=10):
    '''
    :param aim: date
    :param db: existing files
    :param duration: length of days
    :return: data exist or not
    '''
    for hour in iter(range(4)):
        _pd = aim - datetime.timedelta(days=duration, hours=hour)
        _pd_st = 'FY2G_FDI_ALL_NOM_' + _pd.strftime('%Y%m%d_%H%M') + '_' + sub + '.tif'
        if _pd_st in db:
            return db.index(_pd_st)
    return 0


def generate_ts_data(db, x, y, filename, mask, mir, tir, duration=10):
    '''
    :param db: database
    :param x: location indicators | east-west
    :param y: location indicators | north-south
    :param date: should be like 'YYYYMMDD'
    :param path: the output path
    :param duration: the set of previous days | default: 10
    :param hour: aim hour of the day          | default: * (means all day)
    :return:
    '''
    X, Y = 180, 180
    stable_arrays = np.ones((X, Y))
    if np.sum(mask) != X*Y:
        subs = x.__str__().zfill(2) + y.__str__().zfill(2)
        endindex = db.index(filename)
        endtime = datetime.datetime.strptime(re.search(r'\d{8}_\d{4}', filename).group(), '%Y%m%d_%H%M')

        _should_check = -1
        _should_check = check_exist(endtime, db, subs, duration)
        if _should_check:
            # existing_list = np.zeros(24 * duration)
            MIR_DAT = np.zeros((duration, 24, X, Y))
            TIR_DAT = np.zeros((duration, 24, X, Y))
            for _iD in iter(db[_should_check:endindex]):
                if re.search('_\d{2}00_', _iD):
                    idtime = datetime.datetime.strptime(re.search(r'\d{8}_\d{4}', _iD).group(), '%Y%m%d_%H%M')
                    dataid_day = duration - (endtime - idtime).days
                    dataid_day = dataid_day if idtime.hour == endtime.hour else dataid_day - 1
                    dataid_hour = (idtime-endtime).seconds/3600
                    g = gdal.Open(_iD)
                    try:
                        MIR_DAT[dataid_day, dataid_hour, :, :] = g.GetRasterBand(4).ReadAsArray()
                        TIR_DAT[dataid_day, dataid_hour, :, :] = g.GetRasterBand(1).ReadAsArray()
                    except:
                        continue
                    del g

            _w1 = 2 if x == 0 else 0
            _w2 = X - 2 if x == 12 else X
            _l1 = 2 if y == 0 else 0
            _l2 = Y - 2 if y == 12 else Y

            for _x in iter(range(_w1, _w2)):
                for _y in iter(range(_l1, _l2)):
                    if mask[_x, _y] == 0:
                        # same-moment
                        m, t = mir[_x, _y], tir[_x, _y]
                        data_valid = np.ma.masked_less_equal(MIR_DAT[:, 0, _x, _y], 290)
                        avg_smt = np.mean(data_valid)
                        std_smt = np.std(data_valid)
                        top = avg_smt + 2 * std_smt
                        # 2.58 * std_smt / np.sqrt(np.ma.count(data_valid))
                        ts_req1 = m > top
                        stable_arrays[_x, _y] = 0 if ts_req1 else 1

        else:
            print 'Not enough data to process'

    return stable_arrays


def contextual(mir, tir, a, b):
    m, t = mir[a, b], tir[a, b]
    d = m - t
    Mir = mir[a - 2:a + 3, b - 2:b + 3]
    Tir = tir[a - 2:a + 3, b - 2:b + 3]
    delta_array = Mir - Tir
    validpx = starrays(Mir, 290, 350) == 1
    mir_avg = np.mean(Mir[validpx])
    mir_min = np.min(Mir[validpx])
    delta_avg = np.mean(delta_array[validpx])
    delta_min = np.min(delta_array[validpx])

    step2req1 = (m >= mir_min + 15) and (d >= delta_min + 15)
    step2req2 = (m >= mir_avg + 5) and (d >= delta_avg + 5) and (d >= 15)
    step2req3 = (m >= mir_avg + 2) and (m >= 330) and (d >= 15)
    # step2req4 = (m >= 335) and (d >= 15)

    return step2req1, step2req2, step2req3  #, step2req4

#





# fig2 = plt.figure('fig2')
# dataM = np.ma.masked_less_equal(MIR_DAT[:, 0, 51, 123], 270)
# avg = np.mean(dataM)
# std = np.std(dataM)
# top = avg + 2.58 * std / np.sqrt(duration)
# low = avg - 2.58 * std / np.sqrt(duration)
# plt.axhline(top, linestyle='--', linewidth=5)
# plt.axhline(avg, linewidth=5)
# plt.axhline(low, linestyle='--', linewidth=5)
#
# plt.plot(dataM, c='r')
# fig2.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# fig1 = plt.figure('fig1')
# meanday = np.zeros(24, )
# tmeanday = np.zeros(24, )
# for day in iter(range(duration)):
#     data = np.ma.masked_less_equal(MIR_DAT[day, :, 51, 123], 290)
#     tdata = np.ma.masked_less_equal(TIR_DAT[day, :, 51, 123], 270)
#     plt.plot(data)
#     plt.plot(tdata)
#
# for hour in iter(range(24)):
#     hour_dat = np.ma.masked_less_equal(MIR_DAT[:, hour, 51, 123], 290)
#     thour_dat = np.ma.masked_less_equal(TIR_DAT[:, hour, 51, 123], 270)
#     meanday[hour] = np.mean(hour_dat)
#     std = np.std(hour_dat)
#     top = meanday[hour] + 2.58 * std / np.sqrt(duration)
#     low = meanday[hour] - 2.58 * std / np.sqrt(duration)
#     pts = [[hour, hour], [low, top]]
#     # print pts
#     plt.plot(pts[0], pts[1], linewidth=8, c='royalblue')
#     # plt.scatter(pts[0], pts[1], marker='x', c='b')
#     tmeanday[hour] = np.mean(thour_dat)
#     stdt = np.std(thour_dat)
#     topt = tmeanday[hour] + 2.58 * stdt / np.sqrt(duration)
#     lowt = tmeanday[hour] - 2.58 * stdt / np.sqrt(duration)
#     ptst = [[hour, hour], [lowt, topt]]
#     # print pts
#     plt.plot(ptst[0], ptst[1], linewidth=8, c='crimson')
#
# plt.plot(meanday, linewidth=8, c='royalblue')
# plt.plot(tmeanday, linewidth=8, c='crimson')
# fig1.show()