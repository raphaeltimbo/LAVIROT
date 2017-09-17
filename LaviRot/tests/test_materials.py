import pytest
from LaviRot.materials import *
from LaviRot.materials import AISI4140
from numpy.testing import assert_allclose


def test_E():
    mat = Material(rho=7850, G_s=80e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_G_s():
    mat = Material(rho=7850, E=203.2e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_Poisson():
    mat = Material(rho=7850, E=203.2e9, G_s=80e9)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_E_G_s_Poisson():
    mat = Material(rho=7850, E=203.2e9, G_s=80e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_specific_material():
    assert_allclose(AISI4140.rho, 7850)
    assert_allclose(AISI4140.E, 203.2e9)
    assert_allclose(AISI4140.G_s, 80e9)
    assert_allclose(AISI4140.Poisson, 0.27)


def test_error_rho():
    with pytest.raises(ValueError) as ex:
        Material(E=203.2e9, G_s=80e9)
    assert 'Density (rho) not provided.' == str(ex.value)


def test_error_E_G_s_Poisson():
    with pytest.raises(ValueError) as ex:
        Material(rho=785, E=203.2e9)
    assert 'At least 2 arguments from E' in str(ex.value)


def test_oil():
    vg32 = Oil(name='ISO VG32', t_a=40, rho_a=856.8, mu_a=0.0255768159199483,
               t_b=100, mu_b=0.0042050707290448133)

    assert_allclose(vg32.rho_b, 817.72992)
    assert_allclose(vg32.v_a, 2.98515591969518e-05)
    assert_allclose(vg32.v_b, 5.142371125474794e-06)

    t2 = 50
    assert_allclose(vg32.rho(t2), 850.2883199999)
    assert_allclose(vg32.v(t2), 2.2267241505525717e-05)
    assert_allclose(vg32.mu(t2), 0.01893357537076773)

