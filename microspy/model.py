import numpy as np


def sensors_to_modes(acc_1, acc_2):

    acc_c = (acc_1 + acc_2) / 2
    acc_d = (acc_1 - acc_2) / 2

    return acc_c, acc_d

def modes_to_sensors(acc_c, acc_d):

    acc_1 = acc_c + acc_d
    acc_2 = acc_c - acc_d

    return acc_1, acc_2
