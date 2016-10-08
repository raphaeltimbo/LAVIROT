import numpy as np


class BeamElement(object):
    """A beam element.

    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:
    [u1, v1, theta1, psi1, u2, v2, theta2, psi2].T
    Where theta1 and theta2 are the bending on the yz plane and
    psi1 and psi2 are the bending on the xz plane.

    Parameters
    ----------
    n: int
        Element number
    x1: float
        Position of the element first node
    le: float
        Element length
    i_d: float3
        Inner diameter of the element
    o_d: float
        Outer diameter of the element
    E: float
        Young's modulus
    G: float
        Shear modulus
    rho: float
        Density
    axial_force: float
        Axial force
    torque: float
        Torque
    sheaf_effects: bool
        Determine if shear effects are taken into account.
        Default is False
    rotary_inertia: bool
        Determine if rotary_inertia effects are taken into account.
        Default is False
    gyroscopic: bool
        Determine if gyroscopic effects are taken into account.
        Default is False


    Returns
    ----------
    A beam element.

    Examples:

    """

    def __init__(self, n, x1, le, i_d, o_d, E, G, rho,
                 axial_force=0, torque=0,
                 shear_effects=False,
                 rotary_inertia=False,
                 gyroscopic=True):

        self.n = n
        self.x1 = x1
        self.le = le
        self.i_d = i_d
        self.o_d = o_d
        self.E = E
        self.G = G
        self.poisson = 0.5*(E/G) - 1
        self.rho = rho
        self.A = np.pi*(o_d**2 - i_d**2)/4
        #  Ie is the second moment of area of the cross section about
        #  the neutral plane Ie = pi*r**2/4
        self.Ie = np.pi*(o_d**4 - i_d**4)/64
        self.phi = 0

        if shear_effects:
            #  Shear coefficient (phi)
            r = i_d/o_d
            r2 = r*r
            r12 = (1 + r2)**2
            #  kappa as per Hutchinson (2001)
            #kappa = 6*r12*((1+self.poisson)/
            #           ((r12*(7 + 12*self.poisson + 4*self.poisson**2) +
            #             4*r2*(5 + 6*self.poisson + 2*self.poisson**2))))
            kappa = 6*r12*((1+self.poisson)/
                       ((r12*(7 + 6*self.poisson) +
                         r2*(20 + 12*self.poisson))))
            print(kappa)
            self.phi = 12*E*self.Ie/(G*kappa*self.A*self.le**2)

        #  Mass Matrix
        m01 = 312 + 588*self.phi + 280*self.phi**2
        m02 = (44 + 77*self.phi + 35*self.phi**2)*self.le
        m03 = 108 + 252*self.phi + 140*self.phi**2
        m04 = -(26 + 63*self.phi + 35*self.phi**2)*self.le
        m05 = (8 + 14*self.phi + 7*self.phi**2)*self.le**2
        m06 = -(6 + 14*self.phi + 7*self.phi**2)*self.le**2

        M0e = np.array([[m01,     0,     0,   m02,   m03,     0,     0,   m04],
                        [  0,   m01,  -m02,     0,     0,   m03,  -m04,     0],
                        [  0,  -m02,   m05,     0,     0,   m04,   m06,     0],
                        [m02,     0,     0,   m05,  -m04,     0,     0,   m06],
                        [m03,     0,     0,  -m04,   m01,     0,     0,  -m02],
                        [  0,   m03,   m04,     0,     0,   m01,   m02,     0],
                        [  0,  -m04,   m06,     0,     0,   m02,   m05,     0],
                        [m04,     0,     0,   m06,  -m02,     0,     0,   m05]])

        self.M0e = self.rho * self.A * self.le * M0e/(840*(1 + self.phi)**2)

        if rotary_inertia:
            ms1 = 36;
            ms2 = (3 - 15*self.phi)*self.le
            ms3 = (4 + 5*self.phi + 10*self.phi**2)*self.le**2
            ms4 = (-1 - 5*self.phi + 5*self.phi**2)*self.le**2
            Ms = np.array([[ms1,      0,     0,   ms2,  -ms1,     0,     0,   ms2],
                           [   0,   ms1,  -ms2,     0,     0,  -ms1,  -ms2,     0],
                           [   0,  -ms2,   ms3,     0,     0,   ms2,   ms4,     0],
                           [ ms2,     0,     0,   ms3,  -ms2,     0,     0,   ms4],
                           [-ms1,     0,     0,  -ms2,   ms1,     0,     0,  -ms2],
                           [   0,  -ms1,   ms2,     0,     0,   ms1,   ms2,     0],
                           [   0,  -ms2,   ms4,     0,     0,   ms2,   ms3,     0],
                           [ ms2,     0,     0,   ms4,  -ms2,     0,     0,   ms3]])

            Ms = self.rho*self.Ie*Ms/(30*self.le*(1 + self.phi)**2)
            self.M0e = self.M0e + Ms

        #  Stifness matrix



        #  TODO Montar matriz de rigidez