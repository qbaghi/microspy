# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2019
import numpy as np
from .cmsread import read_cmsm


def load_acceleration_modes(file_path, axes=np.array(['X', 'Y', 'Z']),
                            voie='sca', prefix='Acc', datatype='412'):

    acc_d = []
    acc_c = []

    for ax in axes:

        fname = file_path + prefix + 'Dif' + ax + voie + '.bin'
        t, acc_di, mask = read_cmsm(fname, datatype=datatype)
        fname = file_path + prefix + 'Com' + ax + voie + '.bin'
        t, acc_ci, mask = read_cmsm(fname, datatype=datatype)

        acc_d.append(acc_di)
        acc_c.append(acc_ci)

    return np.array(acc_d).T, np.array(acc_c).T, t, mask


def load_acceleration_per_sensor(file_path, axes=np.array(['X', 'Y', 'Z']),
                                 prefix='Acceleration', datatype='412'):

    acc_1 = []
    acc_2 = []

    for ax in axes:

        fname = file_path + 'IS1/' + prefix + ax + '.bin'
        t, acc_1i, mask = read_cmsm(fname, datatype=datatype)
        fname = file_path + 'IS2/' + prefix + ax + '.bin'
        t, acc_2i, mask = read_cmsm(fname, datatype=datatype)

        acc_1.append(acc_1i)
        acc_2.append(acc_2i)

    return np.array(acc_1).T, np.array(acc_2).T, t, mask


def read_gravity_file(file_path, file_name, data_type, axis):
    """
    Function that reads SIMULA simulation files containing 7 columns  :
    date acc_grav_x acc_grav_y acc_grav_z grad_grav_xx grad_grav_xy grad_grav_xz
    grad_grav_yx grad_grav_yy
    grad_grav_yz grad_grav_zx grad_grav_zy grad_grav_zz trace

    The target files are commonly named gravite.dat

    @param file_path : path of the file (without / at the end)
    @type file_path : string
    @param file_name : name of the file
    @type file_name : string
    @param data_type : chosen data among 'acceleration', 'gradient'
    @type data_type : string
    @param axis : chosen axis among X,Y,Z if data_type is 'acceleration'
    or XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ,trace
    if data_type is 'gradient'. Axis may be a vector of strings, e.g.
    axis = [ XX,XY,XZ ]
    @type axis : string

    @return:
        t : N - vector containing times
        data : N x k - matrix containing values of the specified data type and axis (N is the time series length,
        k is the desired number of axis)
    """

    data = np.genfromtxt(file_path
                         + '/'
                         + file_name,
                         names=['dateJour', 'dateSec',
                                'acceleration_x', 'acceleration_y',
                                'acceleration_z',
                                'gradient_xx', 'gradient_xy', 'gradient_xz',
                                'gradient_yy', 'gradient_yz', 'gradient_zz'])

    date_format = 'dateSec'
    n = len(data[date_format])
    m = len(axis)

    d = np.zeros((n, m))
    for i in range(len(axis)):
        d[:, i] = data[data_type + '_' + axis[i]]

    return data[date_format], d


def read_vit_acc_ang_file(file_path,file_name,data_type,axis) :
    """
    Function that reads SIMULA simulation vit_acc_ang files containing 7 columns  :
    date, angular velocities (x,y,z), angular accelerations (x,y,z)

    The target files are commonly named vit_acc_ang_ep.dat

    @param file_path : path of the file (without / at the end)
    @type file_path : string
    @param file_name : name of the file
    @type file_name : string
    @param data_type : chosen data among 'omega', 'omega_dot'
    @type data_type : string
    @param axis : chosen axis among x,y,z
    @type axis : string

    @return:
        t : N - vector containing times
        data : N - vector containing values of the specified data type and axis
    """

    data = np.genfromtxt(file_path + '/' + file_name,
                         names=['dateJour', 'dateSec', 'omega_x', 'omega_y',
                                'omega_z', 'omega_dot_x', 'omega_dot_y',
                                'omega_dot_z'])
    n = len(data['dateSec'])
    m = len(axis)

    d = np.zeros((n, m))

    for i in range(len(axis)):
        d[:, i] = data[data_type + '_' + axis[i]]

    return data['dateSec'], d
