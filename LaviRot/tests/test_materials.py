import pytest
from LaviRot.materials import *
from numpy.testing import assert_allclose


def test_E():
    mat = Material(rho=785, G_s=80e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_G_s():
    mat = Material(rho=785, E=203.2e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_Poisson():
    mat = Material(rho=785, E=203.2e9, G_s=80e9)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_E_G_s_Poisson():
    mat = Material(rho=785, E=203.2e9, G_s=80e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_error_rho():
    with pytest.raises(ValueError) as ex:
        mat = Material(E=203.2e9, G_s=80e9)
    assert 'Density (rho) not provided.' == str(ex.value)


def test_error_E_G_s_Poisson():
    with pytest.raises(ValueError) as ex:
        mat = Material(rho=785, E=203.2e9)
    assert 'At least 2 arguments from E' in str(ex.value)
