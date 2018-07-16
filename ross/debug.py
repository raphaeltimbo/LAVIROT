import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import pint
import lavirot as lr

from copy import copy
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm

dis_folder = '/home/raphael/dissertacao/'
folder = dis_folder + '/modelo_xltrc/analise_xltrc2/injection_bcl306c_ge_brg_coefs/'

imgs = dis_folder + '/seminario/imgs/'
rotor_file = folder + 'bcl306c_rotor_v1e_alterado.xls'
rotor_file_eig = folder + 'bcl306c_rotor_v1e_alterado_eig.xls'
rotor_file_level1 = folder + 'bcl306c_rotor_v1e_alterado_level1.xls'
rotor_sheet = 'Model'

bearing_te_file = folder + 'bcl306c_brg_te_maxclr.xls'
bearing_nte_file = folder + 'bcl306c_brg_te_maxclr.xls'
geseal_file = folder + 'bcl306c_honeycomb.xls'
isotseal_file = folder + 'bcl306c_honeycomb_iso.xls'

shaft = lr.ShaftElement.load_from_xltrc(rotor_file)

bearing0 = lr.BearingElement.load_from_xltrc(8, bearing_nte_file)
bearing1 = lr.BearingElement.load_from_xltrc(49, bearing_te_file)
bearings = [bearing0, bearing1]

disks = lr.LumpedDiskElement.load_from_xltrc(rotor_file)

rotor = lr.Rotor(shaft, disks, bearing_seal_elements=bearings, rated_w=1152, n_eigen=16)

isotseal = lr.SealElement.load_from_xltrc(n=20, file=isotseal_file)


# from 0 to isotseal
def seal_inter2(fac):
    w = isotseal.w
    Kxx_seal_iso = fac * isotseal.kxx.interpolated(w)
    Cxx_seal_iso = fac * isotseal.cxx.interpolated(w)

    seal_interpol2 = lr.SealElement(20, kxx=Kxx_seal_iso, cxx=Cxx_seal_iso, w=w)

    return seal_interpol2


speed = np.linspace(550, 1.3e3, 30)

shaft1 = copy(rotor.shaft_elements)
disks1 = copy(rotor.disk_elements)
bearings1 = copy(rotor.bearing_seal_elements)

seal1 = seal_inter2(1.)
bearings1.append(seal1)

rotor1 = lr.Rotor(shaft1, disks1, bearings1)
res = rotor1.campbell(speed)
res.plot()