import pytest
from LaviRot.rotor import *
from LaviRot.results import *
import numpy as np
from numpy.testing import assert_allclose


@pytest.fixture
def rotor1():
    return rotor_example()


def test_campbell(rotor1):
    speed = np.linspace(0, 400, 100)

    points = campbell(rotor1, speed_rad=speed, plot=False)
    wd_speed0 = np.array([82.7,   86.7,  254.5,  274.3,  679.5,  716.8])

    assert_allclose(points[:, 0], wd_speed0, rtol=1e-3)
