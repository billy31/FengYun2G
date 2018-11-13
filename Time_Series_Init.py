#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-1 下午9:59
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : Time_Series_Init.py
# @Software: PyCharm

SUB = 13
WID = 2288
LEN = 2288
BDS = 6
l = LEN / SUB + 4
w = WID / SUB + 4

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

def generate_file_name(x, y, date, hour, subfix=''):
    subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    _H = hour.__str__().zfill(2) + '00'
    datestring_process = date + '_' + _H
    subfix = '' if subfix == '' else '_' + subfix
    name = 'FY2G_FDI_ALL_NOM_' + datestring_process + '_' + subs + subfix + '.tif'
    return name


def get_stable_array(begin, end, db, x, y, Hour, duration):
    aim_T0 = db[end].split('_')[4]+'_'+db[end].split('_')[5]
    aim_T1 = datetime.datetime.strptime(aim_T0, '%Y%m%d_%H%M')
    total_dlt = datetime.timedelta(days=duration)
    # existing_list = np.zeros(24 * duration)
    mir = np.zeros((24 * duration, w, l))
    tir = np.zeros((24 * duration, w, l))
    for _iD in iter(db[begin:end]):
        obj_T0 = _iD.split('_')[4]+'_' + _iD.split('_')[5]
        obj_T1 = datetime.datetime.strptime(obj_T0, '%Y%m%d_%H%M')
        dlt_T = aim_T1 - obj_T1
        nums_id = (total_dlt - dlt_T).days * 24 + (total_dlt - dlt_T).seconds / 3600
        # existing_list[nums_id] = 1
        g = gdal.Open(_iD)
        tir[nums_id, :, :] = g.GetRasterBand(1).ReadAsArray()
        mir[nums_id, :, :] = g.GetRasterBand(4).ReadAsArray()
        del g

    _w1 = 2 if x == 0 else 0
    _w2 = w-2 if x == 12 else w
    _l1 = 2 if y == 0 else 0
    _l2 = l-2 if y == 12 else l
    stable_arrays = np.zeros((w, l))

    for _x in iter(range(_w1, _w2)):
        for _y in iter(range(_l1, _l2)):
            # Mvalue_matrix = np.zeros((15, 24))
            # Tvalue_matrix = np.zeros((15, 24))
            value_mir = mir[:, _x, _y]
            value_tir = tir[:, _x, _y]
            low_end_tir = np.mean(value_tir[np.nonzero(value_tir)])
            stable_arrays[_x, _y] = np.mean(value_mir[np.where(value_tir > low_end_tir)])
            # time_list = []
            # for id_time, non_zero in enumerate(existing_list):
            #     if non_zero and 200 <= value_mir[id_time] <= 350:
            #         time = id_time % 24
            #         if time == Hour:
            #             time_list.append(id_time)
            #         valuex = (value_mir[id_time] - 200) / 10
            #         Mvalue_matrix[int(valuex), time] += 1
            #         valuex = (value_tir[id_time] - 200) / 10
            #         Tvalue_matrix[int(valuex), time] += 1


            # sort_value_frequency = [sum(value_matrix[ranges, :]) for ranges in range(15)]
            # max_frequency = sort_value_frequency.index(max(sort_value_frequency))
            # mean_value1 = np.mean(value_ori[np.logical_and(value_ori >= max_frequency*10+200,
            #                                                value_ori < max_frequency*10+210)])
            #
            # value_BYTIME = np.argmax(value_matrix[:, Hour])
            # time_list_value = value_ori[time_list]
            # mean_value2 = np.mean(time_list_value[np.logical_and(time_list_value >= value_BYTIME*10+200,
            #                                                      time_list_value < value_BYTIME*10+210)])
            #
            # stable_arrays[0, _x, _y] = mean_value1
            # stable_arrays[1, _x, _y] = mean_value2


            # del Mvalue_matrix, Tvalue_matrix
            # for _x in iter(range(w)):
            #     for _y in iter(range(l)):

    # return 1

    return stable_arrays


def check_exist(aim, db, sub, duration=10):
    '''
    :param aim: date
    :param db: existing files
    :param duration: length of days
    :return: data exist or not
    '''
    # datetime.date.strftime()
    _d = datetime.datetime.strptime(aim, '%Y%m%d_%H%M')
    for hour in iter([0, 1, 2]):
        _pd = _d - datetime.timedelta(days=duration, hours=hour)
        _pd_st = 'FY2G_FDI_ALL_NOM_' + _pd.strftime('%Y%m%d_%H%M') + '_' + sub + '.tif'
        if _pd_st in db:
            return db.index(_pd_st)
    return 0


def generate_ts_data(db, x, y, date, duration=10, hour=0):
    '''
    :param x: location indicators | east-west
    :param y: location indicators | north-south
    :param date: should be like 'YYYYMMDD'
    :param path: the output path
    :param duration: the set of previous days | default: 10
    :param hour: aim hour of the day          | default: * (means all day)
    :return:
    '''
    total_list = db
    subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    try:
        os.chdir(_in_dir + subs + '/')
    except:
        print "No this directory"
        return
    _H = hour.__str__().zfill(2) + '00'
    _should_check = -1
    to_process = date + '_' + _H
    _should_check = check_exist(to_process, total_list, subs, duration)
    if _should_check >= 0:
        try:
            aim_index = total_list.index(generate_file_name(x, y, date, hour)) - 1
        except Exception as e:
            print 'No this file!'
            print e.__str__()
            return 0
        else:
            stable_array = get_stable_array(_should_check, aim_index+1, total_list, x, y, hour, duration)

            print 'Generated!'
            return stable_array
    else:
        print 'No enough data to process'

    return 0


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

if __name__ == '__main__':

    # _in_dir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing/'
    # _out_dir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing_PFires/'
    _in_dir = '/home6/FY2G/subsets/'
    os.chdir(_in_dir)

    _out_dir = '/home6/FY2G/subsets_stable/'
    if os.path.exists(_out_dir) is False:
        os.mkdir(_out_dir)

    ## single step

    # date = '20151220'
    # hour = '03'
    # x = 7
    # y = 9
    # stable_value_mir = generate_ts_data(x, y, date, _out_dir, duration=10, hour=hour)
    # if type(stable_value_mir) is not np.int:
    #     g = gdal.Open(generate_file_name(x, y, date, hour))
    #     obs_TIR = g.GetRasterBand(1).ReadAsArray()
    #     obs_MIR = g.GetRasterBand(4).ReadAsArray()
    #
    #     value_MO_MS = obs_MIR - stable_value_mir
    #     value_MS_TO = stable_value_mir - obs_TIR
    #
    #     value_X = value_MO_MS - value_MS_TO
    #     print 1
    #
    #
    # # for _x in iter(range(SUB)):
    # #     for _y in iter(range(SUB)):
    # #         subs = _x.__str__().zfill(2) + _y.__str__().zfill(2) + '/'
    # #         os.chdir(_in_dir + subs)
    # #         list_of_aims = sorted(glob.glob("FY2G_FDI***.tif"))
    # #         fygdal = gdal.Open(list_of_aims[0])
    # #         print 1

    # Multiple steps
    begindate = datetime.datetime(2015, 12, 20)
    flag = 1
    hour = 0
    while flag:
        date = begindate + datetime.timedelta(hours=hour)
        print date
        for x in iter(range(13)):
            for y in iter(range(13)):
                print x, y
                subs = x.__str__().zfill(2) + y.__str__().zfill(2)
                os.chdir(_in_dir + subs + '/')
                total_list = sorted(glob.glob("FY2G_FDI*" + subs + ".tif"))
                outdir = _out_dir + subs + '/'
                if os.path.exists(outdir) is False:
                    os.mkdir(outdir)
                stable_value_mir = generate_ts_data(total_list, x, y, date.strftime("%Y%m%d"), duration=10, hour=date.hour)
                if type(stable_value_mir) is np.int:
                    flag = stable_value_mir
                    break
                else:
                    g = gdal.Open(generate_file_name(x, y, date.strftime("%Y%m%d"), hour=date.hour))
                    obs_TIR = g.GetRasterBand(1).ReadAsArray()
                    obs_MIR = g.GetRasterBand(4).ReadAsArray()

                    value_MO_MS = obs_MIR - stable_value_mir
                    value_MS_TO = stable_value_mir - obs_TIR

                    value_X = value_MO_MS - value_MS_TO
                    stablemirName = outdir + \
                                    generate_file_name(x, y, date.strftime("%Y%m%d"), date.hour, 'Stable_MIR')
                    deltaName = outdir + \
                                generate_file_name(x, y, date.strftime("%Y%m%d"), date.hour, 'Delta_MOMS_MSTO')

                    if os.path.exists(stablemirName) is False:
                        _Read_Init.arr2TIFF(stable_value_mir, trans, proj, stablemirName, 1)
                    else:
                        print "Exist!"
                    if os.path.exists(deltaName) is False:
                        _Read_Init.arr2TIFF(value_X, trans, proj, deltaName, 1)
                    else:
                        print "Exist!"
        hour += 1


