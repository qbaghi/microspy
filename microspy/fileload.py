import numpy as np
from microspy.cmsread import read_cmsm


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


def periodogram(x, w):

    return np.abs(fft(x * w))**2 / np.sum(w**2)


def coperiodogram(x, y, w):

    x_fft = fft(x * w)
    y_fft = fft(y * w)

    return np.conj(x_fft) * y_fft / np.sum(w**2)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, ifft

    pyfftw.interfaces.cache.enable()

    # N0 data
    # --------------------------------------------------------------------------
    # Root path
    root_path = '/Users/qbaghi/Documents/MICROSCOPE/data/session_380/N0/'
    # Session name
    session_name = 'Session_380_EPR_V3DFIS2_01_SUREF'
    # Data level
    data_level = 'N0c_01'
    # Sensor unit
    su = 'SUREF'
    # N0 data
    file_path = root_path + session_name + '/' + data_level + '/' + su + '/'
    acc_1, acc_2, t0, mask0 = load_acceleration_per_sensor(
        file_path,
        axes=np.array(['X', 'Y', 'Z']),
        prefix='Acceleration',
        datatype='412')

    # N1 data
    # --------------------------------------------------------------------------
    # Root path
    root_path = '/Users/qbaghi/Documents/MICROSCOPE/data/session_380/N1N2/'
    # Session name
    session_name = 'Session_380_EPR_V3DFIS2_01_SUREF'
    # Data level
    data_level = 'N2a_01_L_SCA_G_0050'
    # Sensor unit
    su = 'SUREF'

    # N2 data
    # --------------------------------------------------------------------------
    file_path = root_path + session_name + '/' + data_level + '/' + su + '/'

    # Load data
    acc_d, acc_c, t, mask = load_acceleration_modes(
        file_path, axes=np.array(['X', 'Y', 'Z']), voie='sca', prefix='Acc')

    # Pre-process
    # --------------------------------------------------------------------------
    # SU1
    mask_1 = np.ones(acc_1.shape[0])
    mask_1[acc_1[:, 0] == 0] = 0
    acc_1_m = acc_1 - np.mean(acc_1[mask_1 == 1], axis=0)
    acc_2_m = acc_2 - np.mean(acc_2[mask_1 == 1], axis=0)

    # Differential mode
    mask_d = np.ones(acc_d.shape[0])
    mask_d[acc_d[:, 0] == 0] = 0
    acc_d_m = acc_d - np.mean(acc_d[mask_d == 1], axis=0)

    # Compute periodograms
    fs = 4.0
    f = np.fft.fftfreq(acc_d.shape[0]) * fs
    idpos = np.where(f > 0)[0]
    w1 = np.hanning(acc_1.shape[0]) * mask_1
    p11x = periodogram(acc_1_m[:, 0], w1)
    p22x = periodogram(acc_2_m[:, 0], w1)
    p12x = coperiodogram(acc_1_m[:, 0], acc_2_m[:, 0], w1)

    wd = np.hanning(acc_d.shape[0]) * mask_d
    pdx = periodogram(acc_d_m[:, 0], wd)

    # Plot periodogram
    fig1, ax1 = plt.subplots(nrows=1, sharex=True, sharey=True)
    # ax1.loglog(f[idpos], np.sqrt(pdx[idpos] / fs), color='black', label='Pdx')
    ax1.loglog(f[idpos], np.sqrt(p11x[idpos]), color='black', label='P11x')
    ax1.loglog(f[idpos], np.sqrt(p22x[idpos]), color='red', label='P22x')
    # ax1.loglog(f[idpos], np.sqrt(np.abs(p12x[idpos])), color='gray',
    #            label='P12x')
    ax1.legend(loc='upper left')
    plt.show()
