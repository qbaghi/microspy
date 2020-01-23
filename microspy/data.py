# -*- coding: utf-8 -*-

# data.apid_data_for_fake

# import ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import os
import numpy
import struct
from copy import copy, deepcopy



class Data(object):
    def __init__(self,dates,values,apid_name,name, mask_file = None):
        self._file_dates=dates.replace('\\','/')
        self._name=name
        self._apid_name=apid_name
        self._values=values
        try:
            self._mask_file = mask_file.replace('\\','/')
        except:
            self._mask_file = mask_file

    def __str__(self):
        base=self._name+'{'+self._apid_name+', '+self._file_dates+','+str(self._mask_file)+','+str(len(self._values))
        if len(self._values) >6:
            return base+': ['+str(self._values[:3])+'...'+str(self._values[-3:])+']}'
        else:
            return base+': '+str(self._values)+'}'

    def __repr__(self):
        return str(self)
    def save(self,f):
        f.write(struct.pack('=i',len(self._file_dates)))
        f.write(self._file_dates.replace('\\','/'))
        if self._mask_file is not None:
            f.write(struct.pack('=i',1))
            f.write(struct.pack('=i',len(self._mask_file)))
            f.write(self._mask_file.replace('\\','/'))
        else:
            f.write(struct.pack('=i',0))
        f.write(struct.pack('=i',len(self._name)))
        f.write(self._name)
        f.write(struct.pack('=i',len(self._apid_name)))
        f.write(self._apid_name)
        f.write(struct.pack('=i',len(self._values)))
        numpy.save(f,self._values)

    def read(cls,f):
        date,mask,name,apid_name,vsize =Data.read_infos(f)
        values=numpy.load(f)
        if values.shape[0]==1:
            values=numpy.atleast_1d(values)
        return Data(date,values,apid_name,name, mask_file = mask)

    read=classmethod(read)

    def read_infos(cls,f):
        size,=struct.unpack('=i', f.read(4))
        date=str(f.read(size))
        hasMask,=struct.unpack('=i', f.read(4))
        if hasMask == 1:
            size,=struct.unpack('=i', f.read(4))
            mask=str(f.read(size))
        else:
            mask = None

        size,=struct.unpack('=i', f.read(4))
        name=str(f.read(size))

        size,=struct.unpack('=i', f.read(4))
        apid_name=str(f.read(size))

        vsize=struct.unpack('=i', f.read(4))

        return date,mask,name,apid_name,vsize

    read_infos=classmethod(read_infos)

    def get_file_dates(self):
        return self._file_dates
    def set_file_dates(self,val):
        self._file_dates=val
    def get_name(self):
        return self._name
    def set_name(self,val):
        self._name=val
    def get_apid_name(self):
        return self._apid_name
    def set_api_name(self,val):
        self._apid_name=val
    def get_values(self):
        return self._values
    def set_values(self,val):
        self._values=val
    def set_mask_file(self,val):
        self._mask_file=val
    def get_mask_file(self):
        return self._mask_file



# auto-test ===================================================================
if __name__ == "__main__": # pragma: no cover
    test=2
    if test ==0:
        f= open("F:/SVN/GammeBranches0.36HeadOfficiel/LPDN/N0TestFake/PHASE_1/Session_2/N0b_S/SUREF/IS1/AccelerationX.bin",'rb')
        data=Data.read(f)
        f= open("F:/SVN/GammeBranches0.36HeadOfficiel/LPDN/N0TestFake/PHASE_1/Session_2/N0b_S/SUREF/IS1/CWriteAccelerationX.bin",'rb')
        data2=Data.read(f)
        c=data.get_values()==data2.get_values()
        print(data)
        print((len(data.get_values()),len(data2.get_values())),"=",c.sum())
    if test ==1:
        from transferts import ms2date,ms2julien
        f= open(r"F:\SVN\GammeBranches0.36HeadOfficiel\LPDN\Tests\TestN0c\N0\Phase_1\Session_4_EPI_0_01_SUREF\N0c_01\datation108.bin",'rb')
        data=Data.read(f)
        print(data.get_values())
        data2= map(ms2date,data.get_values())
        print(data2)

        data3=map(ms2julien,data.get_values())
        print(data3)

    var = numpy.zeros(4,numpy.float64)
    var[0] = 1.1
    var[1] = 1.2
    var[2] = 1.3
    var[3] = 1.4
    print('ecriture : ', var)
    dataVar = Data('fichierDates.bin',var,'000','Xc',None)
    f = open(os.path.join(r'.','test.bin'),'wb')
    dataVar.save(f)
    f.close

    f= open(os.path.join(r'.','test.bin'),'rb')
    dataT = Data.read(f)
    var2 = dataT.get_values()
    print('lecture : ', var2)
    if (numpy.max (var2 - var) < 1e-12):
        print('Ecriture et lecture des données OK')
    else:
        print('Problème sur l\'écriture et la lecture de données ! ! ! ! ! ! !')
