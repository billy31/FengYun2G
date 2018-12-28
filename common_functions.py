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
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap as BP
from Revision_FY2G import _HDF_Init as hdfx
import statsmodels.tsa.kalmanf.kalmanfilter as k

colNames = ['YYYYMMDD', 'HHMM', 'sat', 'lat', 'lon', 'T21', 'T31', 'sample', 'FRP', 'conf', 'type']
# [x, y]
workable = [[0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10],
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 8], [1, 9],
            [2, 1], [2, 2], [2, 3], [2, 4],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
            [4, 0], [4, 1], [4, 2], [4, 3],
            [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
            [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7],
            [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10],
            [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10],
            [9, 0], [9, 1], [9, 2], [9, 4], [9, 5], [9, 6], [9, 7], [9, 9], [9, 9], [9, 10],[9, 11],
            [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11],
            [11, 6], [11, 7], [11, 8], [11, 9], [11, 10],
            [12, 7]]

to_test = [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
           [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
           [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10],
           [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10],
           [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10],
           [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10],
           [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10],
           [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10],
           [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 9], [9, 9], [9, 10],
           [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9],
           ]


WID = int(2288 / 13)
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
lonCenter = 104.5

def kalman_testing(z, begin_P = 1.0, begin_kalman=25, x = 23.5, n_iter = 100, Q = 4e-4,  R = 0.2**2):
    '''

    :param z: 测量值
    :param begin_P: 初始估计方差
    :param begin_kalman: 初始值
    :param Q: 测量时的方差
    :param R: 预测时的的方差
    :param x: 真实value
    :param n_iter: times
    :return:
    '''
    sz = (n_iter,)
    # z = np.random.normal(x,0.2,size=sz)  #
    #allocate apace for arrays
    state_kalman = np.zeros(sz)     # a posteri estimate of x 估计值
    state_pre = np.zeros(sz)     # a priori estimate of x     预测值
    P = np.zeros(sz)             # a posteri error estimate
    Pminus = np.zeros(sz)        # a priori error estimate   系统误差
    K = np.zeros(sz)             # gain or blending factor

    state_kalman[0] = begin_kalman     #
    P[0] = begin_P               #

    for k in range(1,n_iter):
        #time update
        state_pre[k] = state_kalman[k-1]    #根据上一个卡尔曼估计值，直接预测，就是原来的值保持不变
        Pminus[k] = P[k-1] + Q              # 存在预测误差

        K[k] = Pminus[k]/(Pminus[k] + R)  # kalman 增益
        state_kalman[k] = state_pre[k] + K[k]*(z[k] - state_pre[k])   #估计值（权重不一样）
        P[k] = (1-K[k])*Pminus[k]
        # if k==n_iter-1:
        #     pre = state_kalman[-1]
        #     Px = P[k-1] + Q
        #     Kx = Px/(Px+R)
        #     newstate = pre + Kx*(z[-1]-pre)

    return state_kalman


def lonlat2xy(trans, lon, lat):
    x = int((lon - trans[0]) / trans[1])
    y = int((lat - trans[3]) / trans[-1])
    return y, x


def imagexy2geo(trans, imagex, imagey):
    '''
    longitude = 经度
    latitude = 纬度
    '''
    lon = trans[0] + imagey * trans[1]
    lat = trans[3] + imagex * trans[5]
    return lon, lat


def get_related_modisfile(modis14files, fshort):
    timeinfo = datetime.datetime.strptime(fshort, '%Y%m%d_%H%M')
    year, doy, hour, minute = [int(m[7:11]) for m in modis14files], [int(m[11:14]) for m in modis14files], \
                              [int(m[15:17]) for m in modis14files], [int(m[17:19]) for m in modis14files]
    timelist = [datetime.datetime(year[i], 1, 1, hour[i], minute[i]) + datetime.timedelta(doy[i] - 1)
                for i in range(modis14files.__len__())]
    ts = np.array([abs(t - timeinfo) for t in timelist])
    modfiles = np.array(modis14files)[np.where(ts < datetime.timedelta(days=0, hours=1, minutes=0))]
    return modfiles


def geo_is_in(aimpx, aimrec):
    '''
    :param aimpx: [经度 纬度]
    :param aimrec: [左上经度 左上纬度 右下经度 右下纬度]
    :return:
    '''
    if aimrec[0] <= aimpx[0] <= aimrec[2] and aimrec[3] <= aimpx[1] <= aimrec[1]:
        return True
    else:
        return False


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
                print('%2d %2d\t| north:%.3f\tsouth:%.3f\t| west:%.3f\teast:%.3f' % \
                      (x, y, latitude[x1, y1], latitude[x2, y2], longitude[x1, y1], longitude[x2, y2]))


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
    print('MIR AVG %-3.3f/ TIR AVG %-3.3f' % (np.mean(MIR[np.where(req_MIR)]), np.mean(TIR[np.where(req_TIR)])))
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
                    dataid_hour = int((idtime-endtime).seconds/3600)
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
                        data_valid = np.ma.masked_less_equal(MIR_DAT[:, 0, _x, _y], 270)
                        avg_smt = np.mean(data_valid)
                        std_smt = np.std(data_valid)
                        top = avg_smt + 2 * std_smt
                        top2 = np.max(data_valid) + 2
                        # 2.58 * std_smt / np.sqrt(np.ma.count(data_valid))
                        ts_req1 = (m >= top) or (m > top2)
                        stable_arrays[_x, _y] = 0 if ts_req1 else 1

        else:
            print('Not enough data to process')

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
    step2req3 = (m >= mir_avg + 1.5) and (mir_avg >= 318) and (d >= 15)

    return step2req1, step2req2, step2req3  #, step2req4


def drawbkmaps_multi(_in_dir, _out_dir, date_input, hour, xranges, yranges, subplot):
    if date_input.__class__ == str:
        date_inthisalgorithm = date_input
    elif date_input.__class__ == str:
        date_inthisalgorithm = datetime.datetime.strptime(date_input, '%Y%m%d')
    else:
        date_inthisalgorithm = date_input.astype(datetime.datetime).strftime('%Y%m%d')

    os.chdir(_in_dir)
    try:
        data_FULLDISK_name = sorted(glob.glob('*' + date_inthisalgorithm + '_' +
                                              hour.__str__().zfill(2) + '00.hdf'))[0]
        data_FULLDISK = gdal.Open(data_FULLDISK_name)
        data_FULLDISK_subs_MIR = gdal.Open(data_FULLDISK.GetSubDatasets()[9][0])
        data_FULLDISK_subs_CAL = gdal.Open(data_FULLDISK.GetSubDatasets()[3][0])
        mir_FULLDISK_RAW = hdfx.cal_values(data_FULLDISK_subs_CAL.GetRasterBand(1).ReadAsArray(),
                                           data_FULLDISK_subs_MIR.GetRasterBand(1).ReadAsArray())
        sns.heatmap(mir_FULLDISK_RAW[np.min(xranges) * WID: (np.max(xranges) + 1) * WID,
                           np.min(yranges) * WID: (np.max(yranges) + 1) * WID], ax=subplot)
    except:
        print('Error')


def drawbkmaps_single(_in_dir, date_input, hour, X, Y, subplot,
                      aimpx=None, widsize=3, annoate=False, required=None):
    # if xranges.__class__ == int:
    #     xranges = [xranges]
    # if yranges.__class__ == int:
    #     yranges = [yranges]

    if date_input.__class__ == str:
        date_inthisalgorithm = date_input
    elif date_input.__class__ == datetime.datetime:
        date_inthisalgorithm = date_input.strftime("%Y%m%d")
    else:
        date_inthisalgorithm = date_input.astype(datetime.datetime).strftime("%Y%m%d")

    # for x in xranges:
    #     for y in yranges:
    x = X
    y = Y

    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.axis('off')

    cmap = 'RdBu_r'
    subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    os.chdir(_in_dir + subs + '/')
    try:
        fyfile = glob.glob("FY2G_FDI*" + date_inthisalgorithm + '_' + hour.__str__().zfill(2) + "00_*.tif")
        if fyfile:
            g = gdal.Open(fyfile[-1])
            mir = g.GetRasterBand(4).ReadAsArray()
            if aimpx:
                aimpx_x, aimpx_y = aimpx
                mir_to_draw = mir[aimpx_x-widsize:aimpx_x+widsize+1, aimpx_y-widsize:aimpx_y+widsize+1]
                sns.heatmap(mir_to_draw, ax=subplot, cmap=cmap, cbar=False, annot=annoate, fmt='.2f')
            else:
                sns.heatmap(mir, ax=subplot, cmap=cmap, cbar=False)
        else:
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)
            subplot.spines['bottom'].set_visible(False)
            subplot.spines['left'].set_visible(False)
    except:
        print('No file')

    if required:
        return np.average(mir), np.std(mir)


def fengyun_ts_algorithm(_in_dir, _out_dir, date_input, hour, xranges, yranges,
                         duration=30, geoxy=None, nospecial=None, getvalues=False):

    if xranges.__class__ == int:
        xranges = [xranges]
    if yranges.__class__ == int:
        yranges = [yranges]

    if date_input.__class__ == str:
        date_inthisalgorithm = date_input
    elif date_input.__class__ == datetime.datetime:
        date_inthisalgorithm = date_input.strftime("%Y%m%d")
    else:
        date_inthisalgorithm = date_input.astype(datetime.datetime).strftime("%Y%m%d")

    stage1 = ['Masked_01_DYNAMIC_', 'Masked_02_CONTEXTUAL_', 'Masked_03_TIMES_']
    stage2 = ['01_DYNAMIC_', '02_CONTEXTUAL_', '03_TIMES_']
    # hour = 5
    # x = 8
    # y = 9
    # subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    # for date in dates:

    if os.path.exists(_out_dir + date_inthisalgorithm + '/') is False:
        os.mkdir(_out_dir + date_inthisalgorithm + '/')
    _out_dir = _out_dir + date_inthisalgorithm + '/'

    total_firepx = []
    if nospecial:
        total_firepxpd = pd.DataFrame(columns=colNames)
    cmap_mask = 'RdYlBu_r'
    values_of_pxs = []
    for x in xranges:
        for y in yranges:
            fig = plt.figure(2)
            subs = x.__str__().zfill(2) + y.__str__().zfill(2)
            os.chdir(_in_dir + subs + '/')
            total_list = sorted(glob.glob("FY2G_FDI*" + subs + ".tif"))
            if os.path.exists(_out_dir + subs + '/') is False:
                os.mkdir(_out_dir + subs + '/')
            outpath = _out_dir + subs + '/'
            # 1
            fyfile = glob.glob("FY2G_FDI*" + date_inthisalgorithm + '_' + hour.__str__().zfill(2) + "00_*.tif")
            print(fyfile)
            if fyfile:
                g = gdal.Open(fyfile[-1])
                mir = g.GetRasterBand(4).ReadAsArray()
                tir = g.GetRasterBand(1).ReadAsArray()
                vis = g.GetRasterBand(5).ReadAsArray()
                delta = mir - tir

                step1req1 = 1 * np.greater_equal(mir, max([np.mean(mir) + 5, 310]))
                step1req2 = 1 * np.greater_equal(delta, max([np.mean(delta) + 5, 15]))
                step1req3 = 1 * np.less_equal(vis, 20)
                # step1req4 = 1 * np.less_equal(tir, 305)
                step1reqs = step1req1 + step1req2 + step1req3  # + step1req4
                step1mask = 1 * np.less(step1reqs, 3)
                sns.heatmap(mir, mask=step1mask)
                plt.title('MIR dynamic threshold: %-4.2f \n DELTA dynamic threshold: %-4.2f' %
                          (max([np.mean(mir) + 5, 310]), max([np.mean(delta) + 5, 15])))
                stageout_masked = outpath + stage1[0] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'
                stageout_class = outpath + stage2[0] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'
                if os.path.exists(stageout_masked) is False:
                    plt.savefig(stageout_masked, dpi=600)
                if os.path.exists(stageout_class) is False:
                    plt.imsave(stageout_class, step1mask, dpi=600, cmap=cmap_mask)
                plt.clf()
                # print(np.sum(step1mask))
                values_of_pxs.append(np.sum(step1mask))

                # 2
                expelled = [0, 1, 178, 179]
                for a in iter(range(180)):
                    for b in iter(range(180)):
                        if step1mask[a, b] == 1:
                            continue
                        else:
                            if a not in expelled and b not in expelled:
                                step2req1, step2req2, step2req3 = contextual(mir, tir, a, b)
                                if step2req1 or step2req2 or step2req3:  # or step2req4:
                                    step1mask[a, b] = 0
                                else:
                                    step1mask[a, b] = 1
                            else:
                                step1mask[a, b] = 1
                sns.heatmap(mir, mask=step1mask)
                plt.title('After contextual method')
                stageout_masked = outpath + stage1[1] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'
                stageout_class = outpath + stage2[1] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'
                if os.path.exists(stageout_masked) is False:
                    plt.savefig(stageout_masked, dpi=600)
                if os.path.exists(stageout_class) is False:
                    plt.imsave(stageout_class, step1mask, dpi=600, cmap=cmap_mask)
                plt.clf()
                # print(np.sum(step1mask))
                values_of_pxs.append(np.sum(step1mask))

                # 3
                if np.sum(step1mask) == 180 ** 2:
                    # print(0)
                    values_of_pxs.append(180 ** 2)
                    continue
                else:
                    stablearray = generate_ts_data(total_list, x, y, fyfile[-1], step1mask, mir, tir, duration=duration)
                    sns.heatmap(mir, mask=stablearray, annot=True, fmt='.1f')
                    firepts = np.where(stablearray == 0)
                    firecounts = firepts[0].__len__()
                    # dm.drawmaps(subs, mir, firepts, 'gray_r')
                    # print(firecounts)
                    pxleft = 180 ** 2 - firecounts
                    values_of_pxs.append(pxleft)

                    # output
                    if firecounts > 0:
                        plt.title('After ts(same-moment) method\nFire counts: %3d' % int(firecounts))
                        stageout_masked = outpath + stage1[2] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'
                        stageout_class = outpath + stage2[2] + date_inthisalgorithm + hour.__str__().zfill(2) + '_' + subs + '.png'

                        if os.path.exists(stageout_masked) is False:
                            plt.savefig(stageout_masked, dpi=600)
                        if os.path.exists(stageout_class) is False:
                            plt.imsave(stageout_class, stablearray, dpi=600, cmap=cmap_mask)
                        plt.clf()
                        geoname = 'FY2G_ALL_' + subs + '_geo.dat'
                        geog = gdal.Open(geoname)
                        lon = geog.GetRasterBand(1).ReadAsArray() + lonCenter
                        lat = geog.GetRasterBand(2).ReadAsArray()
                        lat = np.ma.masked_equal(lat, 300)
                        lon = np.ma.masked_equal(lon, 300)
                        for i in iter(range(firepts[0].__len__())):
                            locpx, locpy = firepts[0][i], firepts[1][i]
                            lonpx, latpx = lon[locpx, locpy], lat[locpx, locpy]
                            total_firepx.append([lonpx, latpx])
                            if nospecial:
                                insertdata = pd.DataFrame([[date_inthisalgorithm, hour.__str__().zfill(2), 'FENGYUN2G',
                                                           latpx, lonpx, mir[locpx, locpy], tir[locpx, locpy],
                                                           0, 0, 0, 0]], columns=colNames)
                                total_firepxpd = total_firepxpd.append(insertdata)

    if getvalues:
        return values_of_pxs

    # for hour in [7]:
    if nospecial:
        try:
            return total_firepxpd
        except:
            return pd.DataFrame(columns=colNames)
    else:
        if geoxy:
            try:
                return firepts
            except:
                return []

        else:
            return total_firepx


def modis_monthly_data(_in_dir, date):
    os.chdir(_in_dir)
    if date.__class__ == str:
        date_fmt = datetime.datetime.strptime(date, '%Y%m%d')
    elif date.__class__ == datetime.datetime:
        date_fmt = date
    else:
        date_fmt = date.astype(datetime.datetime)
    data_str = sorted(glob.glob('*' + date_fmt.year.__str__() + date_fmt.month.__str__().zfill(2) + '*.txt'))
    colNames = ['YYYYMMDD', 'HHMM', 'sat', 'lat', 'lon', 'T21', 'T31', 'sample', 'FRP', 'conf', 'type']
    aimdata = pd.DataFrame(columns=colNames)
    try:
        data = pd.read_csv(data_str[0])
        begin_row = 0
        keapsearchFlag = True
        while keapsearchFlag:
            str_raw = data.iloc[begin_row].values[0]
            for i in range(4):
                str_raw = str_raw.replace('  ', ' ')
            str_data = str_raw.split(' ')
            str_data_fmt = datetime.datetime.strptime(str_data[0].__str__(), '%Y%m%d')
            begin_row += 1

            if str_data_fmt > date_fmt:
                keapsearchFlag = False
            else:
                if str_data[0] == date_fmt.strftime('%Y%m%d'):
                    inputdata = pd.DataFrame([str_data], columns=colNames)
                    aimdata = aimdata.append(inputdata, ignore_index=True)

            # if

        return aimdata
    except Exception as e:
        print(e.__str__())
        return aimdata


def draw_ranges(x, y, geofile='/home2/FY2G/NOM_ITG_2288_2288(0E0N)_LE/NOM_ITG_2288_2288(0E0N)_LE.dat'):
    if x.__class__ == int:
        x = [x]
    else:
        x = list(x)
    if y.__class__ == int:
        y = [y]
    else:
        y = list(y)

    xmin = min(x)
    xmax = max(x) + 1
    ymin = min(y)
    ymax = max(y) + 1

    geog = gdal.Open(geofile)
    lon = geog.GetRasterBand(1).ReadAsArray() + lonCenter
    lat = geog.GetRasterBand(2).ReadAsArray()
    # lat = np.ma.masked_equal(lat, 300)
    # lon = np.ma.masked_equal(lon, 300+lonCenter)

    LON = lon[xmin * WID:xmax * WID, ymin * WID:ymax * WID]
    LAT = lat[xmin * WID:xmax * WID, ymin * WID:ymax * WID]
    lllon, lllat, urlon, urlat = LON[-1, 0], LAT[-1, 0], LON[0, -1], LAT[0, -1]
    if lllon == 300+lonCenter:
        lllon = np.min(np.ma.masked_greater_equal(LON, 300))
    if lllat == 300:
        lllat = np.min(np.ma.masked_greater_equal(LAT, 300))
    if urlon == 300+lonCenter:
        urlon = np.max(np.ma.masked_greater_equal(LON, 300))
    if urlat == 300:
        urlat = np.max(np.ma.masked_greater_equal(LAT, 300))

    return lllon, lllat, urlon, urlat


def px_bounding(pixel, resolutions, ratio=1.5):

    lonpx, latpx = pixel
    resolutionx, resolutiony = resolutions[0], resolutions[1]
    lllon = lonpx - ratio * resolutionx
    lllat = latpx - ratio * resolutiony
    urlon = lonpx + ratio * resolutionx
    urlat = latpx + ratio * resolutiony

    return lllon, lllat, urlon, urlat


def check_inside(polygon, point):
    flag = False
    if polygon[0] <= point[0] <= polygon[2] and polygon[1] <= point[1] <= polygon[3]:
        flag = True
    return flag


def validation(fypxRange, modfires, modfireflag):
    fyfireflag = False
    for modID, modF in enumerate(modfires):
        if check_inside(fypxRange, modF):
            fyfireflag = True
            modfireflag[modID] = 1
    return modfireflag, fyfireflag




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

def generate_ts_pixel_level(db, filename, px_x, px_y, x, y, duration=30, MIR=True):

    TIR = not MIR
    pixel_data = np.zeros((duration, 24))

    subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    try:
        endindex = db.index(filename)
        endtime = datetime.datetime.strptime(re.search(r'\d{8}_\d{4}', filename).group(), '%Y%m%d_%H%M')
    except:
        filename = glob.glob('*' + filename + '*.tif')[-1]
        endindex = db.index(filename)
        endtime = datetime.datetime.strptime(re.search(r'\d{8}_\d{4}', filename).group(), '%Y%m%d_%H%M')

    _should_check = -1
    _should_check = check_exist(endtime, db, subs, duration)
    if _should_check:
        for _iD in iter(db[_should_check:endindex]):
            if re.search('_\d{2}00_', _iD):
                idtime = datetime.datetime.strptime(re.search(r'\d{8}_\d{4}', _iD).group(), '%Y%m%d_%H%M')
                dataid_day = duration - (endtime - idtime).days
                dataid_day = dataid_day if idtime.hour == endtime.hour else dataid_day - 1
                dataid_hour = int((idtime-endtime).seconds/3600)
                g = gdal.Open(_iD)
                try:
                    if MIR:
                        MIR_DAT = g.GetRasterBand(4).ReadAsArray()
                        pixel_data[dataid_day, dataid_hour] = MIR_DAT[px_x, px_y]
                    if TIR:
                        TIR_DAT = g.GetRasterBand(1).ReadAsArray()
                        pixel_data[dataid_day, dataid_hour] = TIR_DAT[px_x, px_y]
                except:
                    continue
                del g

    else:
        print('Not enough data to process')

    return pixel_data