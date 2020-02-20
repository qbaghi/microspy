# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2019
import numpy as np
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
from . import fileload, cmsread
from . import rdbin
from scipy import interpolate


def sensors_to_modes(acc_1, acc_2):

    acc_c = (acc_1 + acc_2) / 2
    acc_d = (acc_1 - acc_2) / 2

    return acc_c, acc_d


def modes_to_sensors(acc_c, acc_d):

    acc_1 = acc_c + acc_d
    acc_2 = acc_c - acc_d

    return acc_1, acc_2


def periodogram(x, w):

    return np.abs(fft(x * w))**2 / np.sum(w**2)


def coperiodogram(x, y, w):

    x_fft = fft(x * w)
    y_fft = fft(y * w)

    return np.conj(x_fft) * y_fft / np.sum(w**2)


class PhysicsModel:

    def __init__(self, actens_path, su):
        """Class to retrieve and preprocess model data from ACTENS software
        including gravitational and inertial gradient, gravitational
        accelerations, angular accelerations, etc.

        Parameters
        ----------
        actens_path : str
            Path where to find the ACTENS files
        su : str
            Type of sensor unit {'SUEP', 'SUREF'}

        """

        self.actens_path = actens_path
        self.su = su

        self.ttensor = np.array([])
        self.itensor = np.array([])
        self.g = np.array([])

        self.sxx = np.array([])
        self.syy = np.array([])
        self.szz = np.array([])
        self.sxy = np.array([])
        self.sxz = np.array([])
        self.syz = np.array([])

        # Vector of acceleration dates
        self.date = 0
        # Vectors of dates for gravitational data (gradients, grav. acc.)
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0
        self.t5 = 0
        # Dates at which all the data will be synchronized
        self.tsync = 0
        # Vectors of position dates
        self.tpos = 0
        self.n = 0
        self.n_pos = 0

        self.omega = np.array([])
        self.omega_dot = np.array([])
        self.delta_t = np.array([])
        self.delta_t_dot = np.array([])
        self.delta_t_dot_dot = np.array([])

        self.a_full = np.array([])
        self.m_sync = np.array([])

        self.f_TM = 9.88e-3
        self.f_cal_lin = 0.00122847554
        self.fe = 4.0

    def loadgradients(self, actens=True, fileformat='bin',
                      date_type='actens',
                      date_path=None,
                      mask_path=None):

        # ==========================================================================
        # 1. Load Actens files
        # ==========================================================================

        if actens:

            if fileformat == 'txt':

                self.t1, self.g = fileload.read_gravity_file(self.actens_path,
                                                             'actens_'
                                                             + self.SU
                                                             + '.dat',
                                                             'acceleration',
                                                             np.array(['x',
                                                                       'y',
                                                                       'z']))

                self.t2, self.ttensor = fileload.read_gravity_file(
                    self.actens_path, 'actens_' + self.SU + '.dat', 'gradient',
                    np.array(['xx', 'xy', 'xz', 'yy', 'yz', 'zz']))
                self.t4, self.omega = fileload.read_vit_acc_ang_file(
                    self.actens_path, 'vit_acc_ang_'+ self.SU + '.dat', 'omega',
                    np.array(['x', 'y', 'z']))
                self.t5, self.omega_dot = fileload.read_vit_acc_ang_file(
                    self.actens_path, 'vit_acc_ang_' + self.SU + '.dat',
                    'omega_dot',
                    np.array(['x', 'y', 'z']))

            elif fileformat == 'bin':

                self.t1, gx, mask = fileload.rdData(
                    self.actens_path + 'accgrav_'+self.SU+'_X.bin')
                n_data = len(gx)
                del gx
                self.ttensor = np.zeros((n_data, 6))
                self.t2, self.ttensor[:, 0], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_XX.bin')
                self.t2, self.ttensor[:, 1], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_XY.bin')
                self.t2, self.ttensor[:, 2], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_XZ.bin')
                self.t2, self.ttensor[:, 3], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_YY.bin')
                self.t2, self.ttensor[:, 4], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_XZ.bin')
                self.t2, self.ttensor[:, 5], mask = cmsread.rdData(
                    self.actens_path + 'gradgrav_'+self.SU+'_ZZ.bin')

                self.g = np.zeros((n_data, 3))
                self.omega = np.zeros((n_data, 3))
                self.omega_dot = np.zeros((n_data, 3))
                axnames = np.array(['X', 'Y', 'Z'])
                for i in range(3):
                    self.t1, self.g[:, i], mask = cmsread.rdData(
                        self.actens_pat + 'accgrav_'
                        + self.SU + '_' + axnames[i] + '.bin')

                    self.t4, self.omega[:, i], mask = cmsread.rdData(
                        self.actens_path
                        + 'vitang_'+self.SU+'_'+axnames[i]+'.bin')
                    self.t5, self.omega_dot[:, i],

                    mask = cmsread.rdData(self.actens_path
                                          + 'accang_'
                                          + self.SU + '_'
                                          + axnames[i] + '.bin')

        else:

            print('WARNING: gradient files are loaded from N1 data')

            if fileformat == 'bin':

                self.g = np.array([rdbin.read(
                    self.actens_path + 'Acc_gravite'+str(i)+'.bin')._values
                    for i in range(1, 4)]).T
                self.omega = np.array([rdbin.read(
                    self.actens_path + 'Vit_angulaire'+str(i)+'.bin')._values
                    for i in range(1, 4)]).T
                self.omega_dot = np.array([rdbin.read(
                    self.actens_path + 'Acc_angulaire'+str(i)+'.bin')._values
                    for i in range(1, 4)]).T

                if date_path is None:
                    date_path = self.actens_path

                dates = rdbin.read(date_path + 'datation_actens.bin')
                self.t1 = dates._values

                indices = ['11', '12', '13', '22', '23', '33']

                self.ttensor = np.array([rdbin.read(
                    self.actens_path
                    + 'Gradient_gravite'+indices[i]+'.bin')._values
                    for i in range(6)]).T
                # d = rdbin.read(file_path)
                # dates = rdbin.read(date_path)
                # t = dates._values
                # mask = rdbin.read(mask_path)
                # m = mask._values
                # obs = d._values

                # self.t1, gx, mask = cmsread.read_cmsm(self.actens_path
                #                                    + 'Acc_gravite1.bin',
                #                                    datatype=date_type)
                # n_data = len(gx)
                # del gx
                # self.ttensor = np.zeros((n_data, 6))
                # self.t2, self.ttensor[:, 0], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite11.bin',
                #     datatype=date_type)
                # self.t2, self.ttensor[:, 1], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite12.bin',
                #     datatype=date_type)
                # self.t2, self.ttensor[:, 2], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite13.bin',
                #     datatype=date_type)
                # self.t2, self.ttensor[:, 3], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite22.bin',
                #     datatype=date_type)
                # self.t2, self.ttensor[:, 4], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite23.bin',
                #     datatype=date_type)
                # self.t2, self.ttensor[:, 5], mask = cmsread.read_cmsm(
                #     self.actens_path + 'Gradient_gravite33.bin',
                #     datatype=date_type)
                #
                # # Initialization
                # self.g = np.zeros((n_data, 3))
                # self.omega = np.zeros((n_data, 3))
                # self.omega_dot = np.zeros((n_data, 3))
                # for i in range(1, 4):
                #     self.t1, self.g[:, i-1], mask = cmsread.read_cmsm(
                #         self.actens_path + 'Acc_gravite'+str(i)+'.bin',
                #         datatype=date_type)
                #     self.t4, self.omega[:, i-1], mask = cmsread.read_cmsm(
                #         self.actens_path + 'Vit_angulaire'+str(i)+'.bin',
                #         datatype=date_type)
                #     self.t5, self.omega_dot[:, i-1], mask = cmsread.read_cmsm(
                #         self.actens_path + 'Acc_angulaire'+str(i)+'.bin',
                #         datatype=date_type)

            elif fileformat == 'txt':

                print('Text file format not implemented')

        # ======================================================================
        # 2. Construct useful signals for fitting
        # ======================================================================
        n3 = 0
        n_data = len(self.g[:, 0])
        # Inertia gradient
        self.itensor = np.zeros((n_data, 3, 3))
        self.itensor[:, 0, 0] = - self.omega[:, 1]**2 - self.omega[:, 2]**2
        self.itensor[:, 1, 1] = - self.omega[:, 0]**2 - self.omega[:, 2]**2
        self.itensor[:, 2, 2] = - self.omega[:, 0]**2 - self.omega[:, 1]**2
        self.itensor[:, 0, 1] = self.omega[:, 0]*self.omega[:, 1] - self.omega_dot[:, 2]
        self.itensor[:, 0, 2] = self.omega[:, 0]*self.omega[:, 2] + self.omega_dot[:, 1]
        self.itensor[:, 1, 0] = self.omega[:, 0]*self.omega[:, 1] + self.omega_dot[:, 2]
        self.itensor[:, 1, 2] = self.omega[:, 1]*self.omega[:, 2] - self.omega_dot[:, 0]
        self.itensor[:, 2, 0] = self.omega[:, 0]*self.omega[:, 2] - self.omega_dot[:, 1]
        self.itensor[:, 2, 1] = self.omega[:, 1]*self.omega[:, 2] + self.omega_dot[:, 0]

        # T - I
        ti_matrix = np.zeros((n_data, 3, 3))
        ti_matrix[:, 0, 0] = self.ttensor[n3:n3+n_data, 0] - self.itensor[n3:n3+n_data, 0, 0]
        ti_matrix[:, 0, 1] = self.ttensor[n3:n3+n_data, 1] - self.itensor[n3:n3+n_data, 0, 1]
        ti_matrix[:, 0, 2] = self.ttensor[n3:n3+n_data, 2] - self.itensor[n3:n3+n_data, 0, 2]
        ti_matrix[:, 1, 0] = self.ttensor[n3:n3+n_data, 1] - self.itensor[n3:n3+n_data, 1, 0]
        ti_matrix[:, 1, 1] = self.ttensor[n3:n3+n_data, 3] - self.itensor[n3:n3+n_data, 1, 1]
        ti_matrix[:, 1, 2] = self.ttensor[n3:n3+n_data, 4] - self.itensor[n3:n3+n_data, 1, 2]
        ti_matrix[:, 2, 0] = self.ttensor[n3:n3+n_data, 2] - self.itensor[n3:n3+n_data, 2, 0]
        ti_matrix[:, 2, 1] = self.ttensor[n3:n3+n_data, 4] - self.itensor[n3:n3+n_data, 2, 1]
        ti_matrix[:, 2, 2] = self.ttensor[n3:n3+n_data, 5] - self.itensor[n3:n3+n_data, 2, 2]

        self.sxx = ti_matrix[:, 0, 0]
        self.syy = ti_matrix[:, 1, 1]
        self.szz = ti_matrix[:, 2, 2]
        self.sxy = 0.5 * (ti_matrix[:, 0, 1] + ti_matrix[:, 1, 0])
        self.sxz = 0.5 * (ti_matrix[:, 0, 2] + ti_matrix[:, 2, 0])
        self.syz = 0.5 * (ti_matrix[:, 1, 2] + ti_matrix[:, 2, 1])

        del ti_matrix

        # Set the synchronized dates to the gradient dates
        self.tsync = self.t1

    def interpolate(self, t_in, y_in, t_out):

        f_interp = interpolate.interp1d(t_in, y_in, kind='linear')

        return f_interp(t_out)

    def interpolategradient(self):

        # Data length
        n2 = len(self.gammad[:, 0])

        # Invalidities of the gradient
        m_grad = np.ones(n2)
        m_grad[self.Sxx == 0] = 0

        # Interpolate zero values in the gradient
        tN = np.arange(0, n2)/self.fe
        ivalues = self.interpolate(tN[m_grad == 1], self.sxx[m_grad == 1],
                                   tN[m_grad == 0])
        Snew = np.zeros(len(self.sxx))
        Snew[:] = self.sxx
        Snew[m_grad == 0] = ivalues
        self.Sxx = Snew

        return tN

    def synchronize(self,date1,date2):

        # Synchronization of datation
        t1_min = np.min(date1)
        t1_max = np.max(date1)
        t2_min = np.min(date2)
        t2_max = np.max(date2)

        if t1_min <= t2_min :
            t_start = t2_min
        else :
            t_start = t1_min
        if t1_max <= t2_max :
            t_end = t1_max
        else:
            t_end = t2_max

        n_start_1 = np.argmin( np.abs( t_start - date1 ) )
        n_end_1 = np.argmin( np.abs( t_end - date1 ) )
        n_start_2 = np.argmin( np.abs( t_start - date2 ) )
        n_end_2 = np.argmin( np.abs( t_end - date2 ) )

        return n_start_1,n_end_1,n_start_2,n_end_2



    def cutgradient(self,ns,ne):

        # Cut the gradients if needed
        self.g = self.g[ns:ne,:]
        self.Sxx = self.Sxx[ns:ne]
        self.Syy = self.Syy[ns:ne]
        self.Szz = self.Szz[ns:ne]
        self.Sxy = self.Sxy[ns:ne]
        self.Sxz = self.Sxz[ns:ne]
        self.Syz = self.Syz[ns:ne]
        # Cut also the angular rates by assuming that initially they are
        # synchronized with the gradients
        self.Omega = self.Omega[ns:ne, :]
        self.Omega_dot = self.Omega_dot[ns:ne, :]

    def synchronizeall(self, pos=False):

        # Synchronization of acceleration and gradient:
        if np.abs(self.t1[0]/self.date[0]) < 1/900.:
            # If t1 is in milisec and date in sec, then convert t1
            self.date = self.date/1000.
        elif np.abs(self.t1[0]/self.date[0]) > 900.:
            self.t1 = self.t1/1000.
        ns1, ne1, ns2, ne2 = self.synchronize(self.date, self.t1)
        # Cut acceleration if needed
        self.gammad = self.gammad[ns1:ne1, :]
        self.gammac = self.gammac[ns1:ne1, :]
        self.tsync = self.date[ns1:ne1]
        self.date = self.date[ns1:ne1]
        # Cut the gradients if needed
        self.cutgradient(ns2, ne2)
        self.t1 = self.t1[ns2:ne2]

        if pos == True :
            # Synchronization of acceleration and position:
            ns1p, ne1p, ns2p, ne2p = self.synchronize(self.tsync, self.tpos)
            # Cut acceleration if needed
            self.gammad = self.gammad[ns1p:ne1p, :]
            self.gammac = self.gammac[ns1p:ne1p, :]
            self.tsync = self.tsync[ns1p:ne1p]
            self.date = self.date[ns1p:ne1p]
            # Cut the gradients if needed
            self.cutgradient(ns1p, ne1p)
            self.t1 = self.t1[ns2p:ne2p]
            # Cut the positions
            self.Delta_t = self.Delta_t[ns2p:ne2p,:]
            self.Delta_t_dot = self.Delta_t_dot[ns2p:ne2p,:]
            self.Delta_t_dot_dot = self.Delta_t_dot_dot[ns2p:ne2p,:]
            self.tpos = self.tpos[ns2p:ne2p]
