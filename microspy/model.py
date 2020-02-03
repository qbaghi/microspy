import numpy as np
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
from microspy import fileload, cmsread


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

    def loadgradients(self, actens=True, fileformat='bin'):

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

                self.t1, gx, mask = cmsread.read_cmsm(self.actens_path
                                                   + 'Acc_gravite1.bin')
                n_data = len(gx)
                del gx
                self.ttensor = np.zeros((n_data, 6))
                self.t2, self.ttensor[:, 0], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite11.bin')
                self.t2, self.ttensor[:, 1], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite12.bin')
                self.t2, self.ttensor[:, 2], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite13.bin')
                self.t2, self.ttensor[:, 3], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite22.bin')
                self.t2, self.ttensor[:, 4], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite23.bin')
                self.t2, self.ttensor[:, 5], mask = cmsread.read_cmsm(
                    self.actens_path + 'Gradient_gravite33.bin')

                # Initialization
                self.g = np.zeros((n_data, 3))
                self.omega = np.zeros((n_data, 3))
                self.omega_dot = np.zeros((n_data, 3))
                for i in range(1, 4):
                    self.t1, self.g[:, i-1], mask = cmsread.read_cmsm(
                        self.actens_path + 'Acc_gravite'+str(i)+'.bin')
                    self.t4, self.omega[:, i-1], mask = cmsread.read_cmsm(
                        self.actens_path + 'Vit_angulaire'+str(i)+'.bin')
                    self.t5, self.omega_dot[:, i-1], mask = cmsread.read_cmsm(
                        self.actens_path + 'Acc_angulaire'+str(i)+'.bin')

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
