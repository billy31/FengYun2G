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
import matplotlib.pyplot as plt
from Revision_FY2G import _Read_Init


def get_stable_array(begin, end, db, x, y, Hour):
    aim_T0 = db[end].split('_')[4]+'_'+db[end].split('_')[5]
    aim_T1 = datetime.datetime.strptime(aim_T0, '%Y%m%d_%H%M')
    total_dlt = datetime.timedelta(days=10)
    existing_list = np.zeros(240)
    mir = np.zeros((10*24, w, l))
    tir = np.zeros((10*24, w, l))
    for _iD in iter(db[begin:end]):
        obj_T0 = _iD.split('_')[4]+'_' + _iD.split('_')[5]
        obj_T1 = datetime.datetime.strptime(obj_T0, '%Y%m%d_%H%M')
        dlt_T = aim_T1 - obj_T1
        nums_id = (total_dlt - dlt_T).days * 24 + (total_dlt - dlt_T).seconds / 3600
        existing_list[nums_id] = 1
        g = gdal.Open(_iD)
        tir[nums_id, :, :] = g.GetRasterBand(1).ReadAsArray()
        mir[nums_id, :, :] = g.GetRasterBand(4).ReadAsArray()
        del g

    _w1 = 2 if x == 0 else 0
    _w2 = w-3 if x == 12 else w-1
    _l1 = 2 if y == 0 else 0
    _l2 = l-3 if y == 12 else l-1
    stable_arrays = np.zeros((2, w, l))

    for _x in iter(range(_w1, _w2)):
        for _y in iter(range(_l1, _l2)):
            value_matrix = np.zeros((15, 24))
            value_ori = mir[:, _x, _y]
            # value_tir =
            time_list = []
            for id_time, non_zero in enumerate(existing_list):
                if non_zero and 200 <= value_ori[id_time] <= 350:
                    time = id_time % 24
                    if time == Hour:
                        time_list.append(id_time)
                    valuex = (value_ori[id_time] - 200) / 10
                    value_matrix[int(valuex), time] += 1

            sort_value_frequency = [sum(value_matrix[ranges, :]) for ranges in range(15)]
            max_frequency = sort_value_frequency.index(max(sort_value_frequency))
            mean_value1 = np.mean(value_ori[np.logical_and(value_ori >= max_frequency*10+200,
                                                           value_ori < max_frequency*10+210)])

            value_BYTIME = np.argmax(value_matrix[:, Hour])
            time_list_value = value_ori[time_list]
            mean_value2 = np.mean(time_list_value[np.logical_and(time_list_value >= value_BYTIME*10+200,
                                                                 time_list_value < value_BYTIME*10+210)])

            stable_arrays[0, _x, _y] = mean_value1
            stable_arrays[1, _x, _y] = mean_value2

            del value_matrix
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
    _d = datetime.datetime.strptime(aim, '%Y%m%d_%H%M')
    for hour in iter([0, 1, 2]):
        _pd = _d - datetime.timedelta(days=duration, hours=hour)
        _pd_st = 'FY2G_FDI_ALL_NOM_' + _pd.strftime('%Y%m%d_%H%M') + '_' + sub + '.tif'
        if _pd_st in db:
            return db.index(_pd_st)
    return 0


def generate_ts_data(x, y, date, path, duration=10, hour='*'):
    '''
    :param x: location indicators | east-west
    :param y: location indicators | north-south
    :param date: should be like 'YYYYMMDD'
    :param path: the output path
    :param duration: the set of previous days | default: 10
    :param hour: aim hour of the day          | default: * (means all day)
    :return:
    '''
    subs = x.__str__().zfill(2) + y.__str__().zfill(2)
    try:
        os.chdir(_in_dir + subs + '/')
    except:
        print "No this directory"
        return
    hours = iter(range(24)) if hour == '*' else iter([hour])
    total_list = sorted(glob.glob("FY2G_FDI*00_" + subs + ".tif"))
    for H in hours:
        _should_check = -1
        _H = H.__str__().zfill(2) + '00'
        to_process = date + '_' + _H
        _should_check = check_exist(to_process, total_list, subs, duration)
        if _should_check >= 0:
            try:
                aim_index = total_list.index('FY2G_FDI_ALL_NOM_' + to_process + '_' + subs + '.tif') - 1
            except Exception as e:
                print 'No this file!'
                print e.__str__()
            else:
                stable_array = get_stable_array(_should_check, aim_index+1, total_list, x, y, H)
                _Read_Init.arr2TIFF(stable_array[0, :, :], 1)
                print 1
        else:
            print 'No enough data to process'
            continue

    return 1


if __name__ == '__main__':

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
    _in_dir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing/'
    _out_dir = '/media/lzy/TOSHIBA WU FY2G_MERSI_Landsat/FY2G_Testing_PFires/'
    os.chdir(_in_dir)
    generate_ts_data(7, 9, '20160101', _out_dir)
    # for _x in iter(range(SUB)):
    #     for _y in iter(range(SUB)):
    #         subs = _x.__str__().zfill(2) + _y.__str__().zfill(2) + '/'
    #         os.chdir(_in_dir + subs)
    #         list_of_aims = sorted(glob.glob("FY2G_FDI***.tif"))
    #         fygdal = gdal.Open(list_of_aims[0])
    #         print 1


