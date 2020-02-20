# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2019
from .data import Data
import sys
import matplotlib.pyplot as plt
import numpy as np
import os.path


def read(file):
    if os.path.exists(file):
        f = open(file, 'rb')
        tmp = Data.read(f)
        f.close()
        return tmp
    else:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(len(sys.argv))
        sys.exit("Usage: rdbin /full/path/to/binfile")
        #sys.exit("Usage: rdbin /full/path/to/binfile /full/path/to/datation.bin")
    try:
        q = 'SUREF' + (sys.argv[1]).split('SUREF')[1]
    except:
        q = 'SUEP' + (sys.argv[1]).split('SUEP')[1]
    print("########################")
    print("Testing " + q)
    print("########################")
    d = read(sys.argv[1])
    p = os.path.join(*sys.argv[1].split('/')[:-1])
    print("Read date in ", os.path.join('/', p, d._file_dates))
    t = read(os.path.join('/', p, d._file_dates))
    print(d._values)
    print(d._file_dates)

    x = t._values
    y = d._values

    print(x)
    print(np.diff(x))
    print("size t, q:", np.shape(x), np.shape(y))

    plt.plot(x,y)#, marker = 'd')
    plt.xlabel("t [sec]")
    plt.ylabel(q)
    plt.suptitle(q)
    plt.show()
