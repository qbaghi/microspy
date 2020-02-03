#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import os
import rdbin
import numpy as np

def read_lpdn(fname):
    """fname -- full path to data file """
    d = rdbin.read(fname)
    p = os.path.join(os.path.dirname(fname) )#get the data directo
    dates = rdbin.read(os.path.join(os.sep, p, d._file_dates)) #read date file (grabed from the data file, where there's a link to the dates file)
    t = dates._values
    obs = d._values

    mask = rdbin.read(os.path.join(os.sep, p, d._mask_file)) #read date file (grabed from the data file, where there's a link to the dates file)
    M = mask._values

    return t, obs, M

def read_cmsm(fname, datatype='412'):

    d = rdbin.read(fname)

    p = os.path.join(os.path.dirname(fname))  #get the data directory

    success = False

    try:
        obs = d._values
        print("Successful acceleration loading.")
    except AttributeError:
        print("Acceleration file not found.")

    try:
        dates = rdbin.read(os.path.join(os.sep, p, d._file_dates))
        t = dates._values
        print("Successful datation loading.")
        mask = rdbin.read(os.path.join(os.sep, p, d.get_mask_file()))
        M = mask._values
        print("Successful mask loading.")
        success = True
    except AttributeError:
        print("Datation file not found, try another file name...")

    if not success:
        try:
            dates = rdbin.read(os.path.join(os.sep, p, 'datation'+datatype+'.bin'))
            t = dates._values
            print("Successful datation loading.")
            mask = rdbin.read(os.path.join(os.sep, p, 'mask'+datatype+'.bin'))
            M = mask._values
            print("Successful mask loading.")
            success = True
        except AttributeError:
            print("Datation file not found, try another path...")

    if not success:
        try:
            datepath = os.path.abspath(os.path.join(fname , "../../.."))
            dates = rdbin.read(datepath + '/' + 'datation' + datatype + '.bin')
            t = dates._values
            print("Successful datation loading.")
            mask = rdbin.read(datepath + '/' + 'mask' + datatype + '.bin')
            M = mask._values
            print("Successful mask loading.")
            success = True
        except AttributeError:
            print("Datation file not found, try another path...")

    if not success:
        try:
            datepath = os.path.abspath(os.path.join(fname , "../../.."))
            new_fname = datepath + '/DatesAndMasks_01/datation' + datatype + '.bin'
            dates = rdbin.read(new_fname)
            t = dates._values
            print("Successful datation loading.")
            mask = rdbin.read(datepath + '/DatesAndMasks_01/mask' + datatype + '.bin')
            M = mask._values
            print("Successful mask loading.")
            success = True
        except AttributeError:
            print("Datation file not found, try another path...")

    if not success:
        try:
            datepath = os.path.abspath(os.path.join(fname , "../../../.."))
            new_fname = datepath + '/DatesAndMasks_01/datation' + datatype + '.bin'
            dates = rdbin.read(new_fname)
            t = dates._values
            print("Successful datation loading.")
            mask = rdbin.read(datepath+'/DatesAndMasks_01/mask' + datatype + '.bin')
            M = mask._values
            print("Successful mask loading.")
            success = True
        except AttributeError:
            print("Datation loading failed.")

    return t, obs, M


def read_QO(fname,datatype='Acc'):

    d = rdbin.read(fname)

    p = os.path.join(os.path.dirname(fname) )#get the data directory

    try:
        dates = rdbin.read(os.path.join(os.sep, p, d._file_dates))
        t = dates._values
    except AttributeError:
        dates = rdbin.read(os.path.join(os.sep, p, 'datation'+datatype+'.bin'))
        t = dates._values

    obs = d._values

    try:
        mask = rdbin.read(os.path.join(os.sep, p, d.get_mask_file()))
        M = mask._values
    except AttributeError:
        try:
            # go 4 levels before
            maskpath = os.path.abspath(os.path.join(fname ,"../../../.."))
            mask = rdbin.read(maskpath+'/DatesAndMasks_01/mask411.bin')
            M = mask._values
        except AttributeError:
            if '_x' in fname :
                mask = rdbin.read(os.path.join(os.sep, p, 'MaskAccX.bin'))
            elif '_y' in fname :
                mask = rdbin.read(os.path.join(os.sep, p, 'MaskAccY.bin'))
            elif '_z' in fname :
                mask = rdbin.read(os.path.join(os.sep, p, 'MaskAccZ.bin'))
            elif 'Position' in fname :
                mask = rdbin.read(os.path.join(os.sep, p, 'MaskPosition.bin'))
            else:
                print('Mask not found.')
            M = mask._values


    return t, obs, M


def rdData(filename, plotit = False):
    #read in the data file
    d = rdbin.read(filename)

    #get the data directory
    #p = os.path.join(*fullpath_to_binfile.split(os.sep)[:-1])
    p=os.path.dirname(filename)

    #read date file (grabed from the data file, where there's a link to the dates file)
    if '.bin' in os.path.join(os.sep, p, d._file_dates) :
        t = rdbin.read(os.path.join(os.sep, p, d._file_dates))
    else:
        t = rdbin.read(os.path.join(os.sep, p, d._file_dates) + '.bin')

    #grab dates (x) and data (y)
    x = t._values / 1000
    y = d._values

    #read mask
    if d.get_mask_file() != 'none':
        md = rdbin.read(os.path.join(os.sep, p, d.get_mask_file()))
        m = md._values
    else:
        m = np.ones(x.size)

    if plotit:
        plt.plot(x, y)
        plt.xlabel('t [sec]')
        plt.show()

    return x, y, m


def read_actens(fname):
    """fname -- full path to data file """
    d = rdbin.read(fname)
    p = os.path.join(os.path.dirname(fname) )#get the data directo
    dates = rdbin.read(os.path.join(os.sep, p, d._file_dates)) #read date file (grabed from the data file, where there's a link to the dates file)
    t = dates._values
    obs = d._values

    return t, obs