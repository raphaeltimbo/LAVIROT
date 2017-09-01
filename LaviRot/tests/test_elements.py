import pytest
import os
from LaviRot.elements import *
from LaviRot.materials import steel
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

################################################################################
# Shaft tests
################################################################################

test_dir = os.path.dirname(__file__)


@pytest.fixture
def eb():
    #  Euler-Bernoulli element
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05
    return ShaftElement(le_, i_d_, o_d_, steel,
                        shear_effects=False, rotary_inertia=False)


def test_parameters_eb(eb):
    assert eb.phi == 0
    assert eb.L == 0.25
    assert eb.i_d == 0
    assert eb.o_d == 0.05
    assert eb.E == 211e9
    assert eb.G_s == 81.2e9
    assert eb.rho == 7810
    assert_almost_equal(eb.Poisson, 0.29926108)
    assert_almost_equal(eb.A, 0.00196349)
    assert_almost_equal(eb.Ie*1e7, 3.06796157)


def test_mass_matrix_eb(eb):
    M0e_eb = np.array([[ 1.42395,  0.     ,  0.     ,  0.0502 ,  0.49291,  0.     ,  0.     , -0.02967],
                       [ 0.     ,  1.42395, -0.0502 ,  0.     ,  0.     ,  0.49291,  0.02967,  0.     ],
                       [ 0.     , -0.0502 ,  0.00228,  0.     ,  0.     , -0.02967, -0.00171,  0.     ],
                       [ 0.0502 ,  0.     ,  0.     ,  0.00228,  0.02967,  0.     ,  0.     , -0.00171],
                       [ 0.49291,  0.     ,  0.     ,  0.02967,  1.42395,  0.     ,  0.     , -0.0502 ],
                       [ 0.     ,  0.49291, -0.02967,  0.     ,  0.     ,  1.42395,  0.0502 ,  0.     ],
                       [ 0.     ,  0.02967, -0.00171,  0.     ,  0.     ,  0.0502 ,  0.00228,  0.     ],
                       [-0.02967,  0.     ,  0.     , -0.00171, -0.0502 ,  0.     ,  0.     ,  0.00228]])
    assert_allclose(eb.M(), M0e_eb, rtol=1e-3)


def test_stiffness_matrix_eb(eb):
    K0e_eb = np.array([[ 4.97157,  0.     ,  0.     ,  0.62145, -4.97157,  0.     ,  0.     ,  0.62145],
                       [ 0.     ,  4.97157, -0.62145,  0.     ,  0.     , -4.97157, -0.62145,  0.     ],
                       [ 0.     , -0.62145,  0.10357,  0.     ,  0.     ,  0.62145,  0.05179,  0.     ],
                       [ 0.62145,  0.     ,  0.     ,  0.10357, -0.62145,  0.     ,  0.     ,  0.05179],
                       [-4.97157,  0.     ,  0.     , -0.62145,  4.97157,  0.     ,  0.     , -0.62145],
                       [ 0.     , -4.97157,  0.62145,  0.     ,  0.     ,  4.97157,  0.62145,  0.     ],
                       [ 0.     , -0.62145,  0.05179,  0.     ,  0.     ,  0.62145,  0.10357,  0.     ],
                       [ 0.62145,  0.     ,  0.     ,  0.05179, -0.62145,  0.     ,  0.     ,  0.10357]])
    assert_almost_equal(eb.K() / 1e7, K0e_eb, decimal=5)


@pytest.fixture
def tim():
    #  Timoshenko element
    z_ = 0
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05
    return ShaftElement(le_, i_d_, o_d_, steel,
                        rotary_inertia=True,
                        shear_effects=True)


def test_parameters_tim(tim):
    assert_almost_equal(tim.phi, 0.08795566)
    assert_almost_equal(tim.Poisson, 0.29926108)
    assert_almost_equal(tim.A, 0.00196349)
    assert_almost_equal(tim.Ie*1e7, 3.06796157)


def test_mass_matrix_tim(tim):
    M0e_tim = np.array([[ 1.42051,  0.     ,  0.     ,  0.04932,  0.49635,  0.     ,  0.     , -0.03055],
                        [ 0.     ,  1.42051, -0.04932,  0.     ,  0.     ,  0.49635,  0.03055,  0.     ],
                        [ 0.     , -0.04932,  0.00231,  0.     ,  0.     , -0.03055, -0.00178,  0.     ],
                        [ 0.04932,  0.     ,  0.     ,  0.00231,  0.03055,  0.     ,  0.     , -0.00178],
                        [ 0.49635,  0.     ,  0.     ,  0.03055,  1.42051,  0.     ,  0.     , -0.04932],
                        [ 0.     ,  0.49635, -0.03055,  0.     ,  0.     ,  1.42051,  0.04932,  0.     ],
                        [ 0.     ,  0.03055, -0.00178,  0.     ,  0.     ,  0.04932,  0.00231,  0.     ],
                        [-0.03055,  0.     ,  0.     , -0.00178, -0.04932,  0.     ,  0.     ,  0.00231]])
    assert_almost_equal(tim.M(), M0e_tim, decimal=5)


def test_stiffness_matrix_tim(tim):
    K0e_tim = np.array([[ 4.56964,  0.     ,  0.     ,  0.57121, -4.56964,  0.     ,  0.     ,  0.57121],
                        [ 0.     ,  4.56964, -0.57121,  0.     ,  0.     , -4.56964, -0.57121,  0.     ],
                        [ 0.     , -0.57121,  0.09729,  0.     ,  0.     ,  0.57121,  0.04551,  0.     ],
                        [ 0.57121,  0.     ,  0.     ,  0.09729, -0.57121,  0.     ,  0.     ,  0.04551],
                        [-4.56964,  0.     ,  0.     , -0.57121,  4.56964,  0.     ,  0.     , -0.57121],
                        [ 0.     , -4.56964,  0.57121,  0.     ,  0.     ,  4.56964,  0.57121,  0.     ],
                        [ 0.     , -0.57121,  0.04551,  0.     ,  0.     ,  0.57121,  0.09729,  0.     ],
                        [ 0.57121,  0.     ,  0.     ,  0.04551, -0.57121,  0.     ,  0.     ,  0.09729]])
    assert_almost_equal(tim.K() / 1e7, K0e_tim, decimal=5)


def test_gyroscopic_matrix_tim(tim):
    G0e_tim = np.array([[ -0.     ,  19.43344,  -0.22681,  -0.     ,  -0.     , -19.43344,  -0.22681,  -0.     ],
                        [-19.43344,  -0.     ,  -0.     ,  -0.22681,  19.43344,  -0.     ,  -0.     ,  -0.22681],
                        [  0.22681,  -0.     ,  -0.     ,   0.1524 ,  -0.22681,  -0.     ,  -0.     ,  -0.04727],
                        [ -0.     ,   0.22681,  -0.1524 ,  -0.     ,  -0.     ,  -0.22681,   0.04727,  -0.     ],
                        [ -0.     , -19.43344,   0.22681,  -0.     ,  -0.     ,  19.43344,   0.22681,  -0.     ],
                        [ 19.43344,  -0.     ,  -0.     ,   0.22681, -19.43344,  -0.     ,  -0.     ,   0.22681],
                        [  0.22681,  -0.     ,  -0.     ,  -0.04727,  -0.22681,  -0.     ,  -0.     ,   0.1524 ],
                        [ -0.     ,   0.22681,   0.04727,  -0.     ,  -0.     ,  -0.22681,  -0.1524 ,  -0.     ]])
    assert_almost_equal(tim.G() * 1e3, G0e_tim, decimal=5)


################################################################################
# Disk tests
################################################################################

@pytest.fixture
def disk():
    return DiskElement(0, steel, 0.07, 0.05, 0.28)


def test_mass_matrix_disk(disk):
    Md1 = np.array([[ 32.58973,   0.     ,   0.     ,   0.     ],
                    [  0.     ,  32.58973,   0.     ,   0.     ],
                    [  0.     ,   0.     ,   0.17809,   0.     ],
                    [  0.     ,   0.     ,   0.     ,   0.17809]])
    assert_almost_equal(disk.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk(disk):
    Gd1 = np.array([[ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.32956],
                    [ 0.     ,  0.     , -0.32956,  0.     ]])
    assert_almost_equal(disk.G(), Gd1, decimal=5)


def test_errors():
    with pytest.raises(TypeError) as ex:
        DiskElement(1.0, steel, 0.07, 0.05, 0.28)
    assert 'n should be int, not float' == str(ex.value)


################################################################################
# Bearing tests
################################################################################

# TODO add tests for bearing elements
@pytest.fixture
def bearing0():
    Kxx_bearing = np.array([8.5e+07, 1.1e+08, 1.3e+08, 1.6e+08, 1.8e+08,
                           2.0e+08, 2.3e+08, 2.5e+08, 2.6e+08])
    Kyy_bearing = np.array([9.2e+07, 1.1e+08, 1.4e+08, 1.6e+08, 1.9e+08,
                            2.1e+08, 2.3e+08, 2.5e+08, 2.6e+08])
    Cxx_bearing = np.array([226837, 211247, 197996, 185523, 174610,
                            163697, 153563, 144209, 137973])
    Cyy_bearing = np.array([235837, 211247, 197996, 185523, 174610,
                            163697, 153563, 144209, 137973])
    wb = np.array([314.2, 418.9, 523.6, 628.3, 733.,
                   837.8, 942.5, 1047.2, 1151.9])
    bearing0 = BearingElement(4,
                              kxx=Kxx_bearing,
                              kyy=Kyy_bearing,
                              cxx=Cxx_bearing,
                              cyy=Cyy_bearing,
                              w=wb)
    return bearing0


def test_bearing_interpol_kxx(bearing0):
    assert_allclose(bearing0.kxx(314.2), 8.5e7)
    assert_allclose(bearing0.kxx(1151.9), 2.6e8)


def test_bearing_interpol_kyy(bearing0):
    assert_allclose(bearing0.kyy(314.2), 9.2e7)
    assert_allclose(bearing0.kyy(1151.9), 2.6e8)


def test_bearing_interpol_cxx(bearing0):
    assert_allclose(bearing0.cxx(314.2), 226837, rtol=1e5)
    assert_allclose(bearing0.cxx(1151.9), 137973, rtol=1e5)


def test_bearing_interpol_cyy(bearing0):
    assert_allclose(bearing0.kxx(314.2), 235837, rtol=1e5)
    assert_allclose(bearing0.kxx(1151.9), 2.6e8, rtol=1e5)


def test_bearing_error1():
    speed = np.linspace(0, 10000, 5)
    kx = 1e8 * speed
    cx = 1e8 * speed
    with pytest.raises(Exception) as excinfo:
        BearingElement(-1, kxx=kx, cxx=cx)
    assert ('w should be an array with'
            ' the parameters dimension') in str(excinfo.value)


def test_load_shaft_from_xltrc():
    file = os.path.join(test_dir, 'data/xl_rotor.xls')

    shaft = ShaftElement.load_from_xltrc(file)
    assert len(shaft) == 93
    assert_allclose(shaft[0].L, 0.0355)
    assert_allclose(shaft[0].i_d, 0.1409954)
    assert_allclose(shaft[0].o_d, 0.151003)
    assert_allclose(shaft[0].rho, 7833.4128)
    assert_allclose(shaft[0].E, 206842710000.0)
    assert_allclose(shaft[0].Poisson, 0.25)


def test_load_bearing_from_xltrc():
    file = os.path.join(test_dir, 'data/xl_bearing.xls')

    bearing = BearingElement.load_from_xltrc(0, file)

    K0 = np.array([[1.056079e+07, -6.877765e+02],
                   [6.594875e+02,  6.551263e+07]])
    C0 = np.array([[1.881813e+05, -1.512049e-01],
                   [-9.357054e-02, 3.402098e+05]])

    assert_allclose(bearing.w[0], 314.159265)

    assert_allclose(bearing.K(0), K0, rtol=1e-3)
    assert_allclose(bearing.C(0), C0, rtol=1e-3)


def test_load_lumped_disk_from_xltrc():
    file = os.path.join(test_dir, 'data/xl_rotor.xls')

    disks = LumpedDiskElement.load_from_xltrc(file)
    disk1_M = np.array([[ 6.909992,  0.      ,  0.      ,  0.      ],
                        [ 0.      ,  6.909992,  0.      ,  0.      ],
                        [ 0.      ,  0.      ,  0.025   ,  0.      ],
                        [ 0.      ,  0.      ,  0.      ,  0.025   ]])

    assert_allclose(disks[1].M(), disk1_M, rtol=1e-4)


