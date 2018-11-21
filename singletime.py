#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-21 下午8:59
# @Author  : Zhengyang Lin
# @Site    : 
# @File    : singletime.py
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
from common_functions import arr2TIFF
import pandas as pd
from common_functions import starrays



if __name__ == '__main__':
    inputs = sorted(glob.glob('/home/lzy/pics/FY2G_FDI_ALL_NOM_20160106*.tif'))
    fygdal = gdal.Open(inputs[0])
    trans = fygdal.GetGeoTransform()
    proj = fygdal.GetProjection()
    del fygdal
    for inputfile in inputs:
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

        req_MIR_new = 1 * np.greater_equal(MIR, np.mean(MIR[req_MIR]))
        req_TIR_new = 1 * np.greater_equal(TIR, np.mean(TIR[req_TIR]))
        req_IR = 1 * np.equal(req_MIR_new + req_TIR_new, 2)
        reqs_dynamic = 1 * np.equal(req_IR + reqs_basic, 2)
        locs = np.where(reqs_dynamic == True)
        length = np.sum(reqs_dynamic)
        if length > 0:
            outfile = np.zeros((180, 180))
            for i in iter(range(length)):
                x, y = locs[0][i] + 2, locs[1][i] + 2
                mir_array = MIR_original[x-2:x+3, y-2:y+3]
                tir_array = TIR_original[x-2:x+3, y-2:y+3]
                delta_array = mir_array - tir_array
                validpx = starrays(mir_array, 290, 350) == 1
                mir_avg = np.mean(mir_array[validpx])
                mir_std = np.std(mir_array[validpx])
                delta_avg = np.mean(delta_array[validpx])
                delta_std = np.std(delta_array[validpx])
                if MIR_original[x, y] > mir_avg + 0.5 * mir_std and \
                    DELTA_original[x, y] > delta_avg + 0.5 * delta_std:
                    outfile[x, y] = 1
            output = inputfile[:-4] + '_SINGLE.png'

            print "%s\t original: 32400/after : %4d: reduced by %.2f %%" % (inputfile.split('/')[-1],
                                                                            np.sum(outfile),
                                                                            100*(1 - float(np.sum(outfile)) / 32400))
            # plt.imsave(output, outfile)
            # arr2TIFF(outfile, trans, proj, output, 1)
            # print output
        else:
            continue








