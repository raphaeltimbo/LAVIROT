import numpy as np
import scipy.interpolate as interpolate
from itertools import permutations


__all__ = ["ShaftElement", "DiskElement", "BearingElement", "SealElement"]


class ShaftElement(object):
    r"""A shaft element.

    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:

    .. math:: [x_1, y_1, \alpha_1, \beta_1, x_2, y_2, \alpha_2, \beta_2]^T

    Where :math:`\alpha_1` and :math:`\alpha_2` are the bending on the yz plane and
    :math:`\beta_1` and :math:`\beta_2` are the bending on the xz plane.

    Parameters
    ----------
    n : int
        Element number (coincident with it's first node).
    L : float
        Element length.
    i_d : float
        Inner diameter of the element.
    o_d : float
        Outer diameter of the element.
    E : float
        Young's modulus.
    G_s : float
        Shear modulus.
    rho : float
        Density.
    axial_force : float
        Axial force.
    torque : float
        Torque.
    shear_effects : bool
        Determine if shear effects are taken into account.
        Default is False.
    rotary_inertia : bool
        Determine if rotary_inertia effects are taken into account.
        Default is False.
    gyroscopic : bool
        Determine if gyroscopic effects are taken into account.
        Default is False.

    Returns
    -------

    Attributes
    ----------
    poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane Ie = pi*r**2/4
    phi : float
        Constant that is used according to [1]_ to consider rotary
        inertia and shear effects. If these are not considered phi=0.

    References
    ----------
    .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
       & AW Lees, published by Cambridge University Press, 2010 pp. 158-166.

    Examples
    --------
    >>> n = 1
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> E = 211e9
    >>> G_s = 81.2e9
    >>> rho = 7810
    >>> Euler_Bernoulli_Element = ShaftElement(n, le, i_d, o_d, E, G_s, rho)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> Timoshenko_Element = ShaftElement(n, le, i_d, o_d, E, G_s, rho,
    ...                                   rotary_inertia=True,
    ...                                   shear_effects=True)
    >>> Timoshenko_Element.phi
    0.08795566502463055
    """
    #  TODO detail this class attributes inside the docstring
    #  TODO add __repr__ to the class
    def __init__(self, n, L, i_d, o_d, E, G_s, rho,
                 axial_force=0, torque=0,
                 shear_effects=False,
                 rotary_inertia=False,
                 gyroscopic=True):

        self.shear_effects = shear_effects
        self.rotary_inertia = rotary_inertia
        self.gyroscopic = gyroscopic

        self.n = n
        self.L = L
        self.i_d = i_d
        self.o_d = o_d
        self.E = E
        self.G_s = G_s
        self.poisson = 0.5*(E/G_s) - 1
        self.rho = rho
        self.A = np.pi*(o_d**2 - i_d**2)/4
        #  Ie is the second moment of area of the cross section about
        #  the neutral plane Ie = pi*r**2/4
        self.Ie = np.pi*(o_d**4 - i_d**4)/64
        phi = 0

        if shear_effects:
            #  Shear coefficient (phi)
            r = i_d/o_d
            r2 = r*r
            r12 = (1 + r2)**2
            #  kappa as per Hutchinson (2001)
            #kappa = 6*r12*((1+self.poisson)/
            #           ((r12*(7 + 12*self.poisson + 4*self.poisson**2) +
            #             4*r2*(5 + 6*self.poisson + 2*self.poisson**2))))
            #  kappa as per Cowper (1996)
            kappa = 6*r12*((1+self.poisson)/
                       ((r12*(7 + 6*self.poisson) +
                         r2*(20 + 12*self.poisson))))
            phi = 12*E*self.Ie/(G_s*kappa*self.A*L**2)

        self.phi = phi

    def M(self):
        r"""Mass matrix for an instance of a shaft element.

        Returns
        -------
        Mass matrix for the shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(1, 0.25, 0, 0.05, 211e9, 81.2e9, 7810,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        phi = self.phi
        L = self.L

        m01 = 312 + 588*phi + 280*phi**2
        m02 = (44 + 77*phi + 35*phi**2)*L
        m03 = 108 + 252*phi + 140*phi**2
        m04 = -(26 + 63*phi + 35*phi**2)*L
        m05 = (8 + 14*phi + 7*phi**2)*L**2
        m06 = -(6 + 14*phi + 7*phi**2)*L**2

        M = np.array([[m01,     0,     0,   m02,   m03,     0,     0,   m04],
                      [  0,   m01,  -m02,     0,     0,   m03,  -m04,     0],
                      [  0,  -m02,   m05,     0,     0,   m04,   m06,     0],
                      [m02,     0,     0,   m05,  -m04,     0,     0,   m06],
                      [m03,     0,     0,  -m04,   m01,     0,     0,  -m02],
                      [  0,   m03,   m04,     0,     0,   m01,   m02,     0],
                      [  0,  -m04,   m06,     0,     0,   m02,   m05,     0],
                      [m04,     0,     0,   m06,  -m02,     0,     0,   m05]])

        M = self.rho * self.A * self.L * M/(840*(1 + phi)**2)

        if self.rotary_inertia:
            ms1 = 36
            ms2 = (3 - 15*phi)*L
            ms3 = (4 + 5*phi + 10*phi**2)*L**2
            ms4 = (-1 - 5*phi + 5*phi**2)*L**2
            Ms = np.array([[ms1,      0,     0,   ms2,  -ms1,     0,     0,   ms2],
                           [   0,   ms1,  -ms2,     0,     0,  -ms1,  -ms2,     0],
                           [   0,  -ms2,   ms3,     0,     0,   ms2,   ms4,     0],
                           [ ms2,     0,     0,   ms3,  -ms2,     0,     0,   ms4],
                           [-ms1,     0,     0,  -ms2,   ms1,     0,     0,  -ms2],
                           [   0,  -ms1,   ms2,     0,     0,   ms1,   ms2,     0],
                           [   0,  -ms2,   ms4,     0,     0,   ms2,   ms3,     0],
                           [ ms2,     0,     0,   ms4,  -ms2,     0,     0,   ms3]])

            Ms = self.rho * self.Ie * Ms/(30*L*(1 + phi)**2)
            M = M + Ms

        return M

    def K(self):
        r"""Stiffness matrix for an instance of a shaft element.

        Returns
        -------
        Stiffness matrix for the shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(1, 0.25, 0, 0.05, 211e9, 81.2e9, 7810,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.K()[:4, :4]/1e6
        array([[ 45.69644273,   0.        ,   0.        ,   5.71205534],
               [  0.        ,  45.69644273,  -5.71205534,   0.        ],
               [  0.        ,  -5.71205534,   0.97294287,   0.        ],
               [  5.71205534,   0.        ,   0.        ,   0.97294287]])
        """
        phi = self.phi
        L = self.L

        K = np.array([[12,     0,            0,          6*L,  -12,     0,            0,          6*L],
                        [0,     12,         -6*L,            0,    0,   -12,         -6*L,            0],
                        [0,   -6*L, (4+phi)*L**2,            0,    0,   6*L, (2-phi)*L**2,            0],
                        [6*L,    0,            0, (4+phi)*L**2, -6*L,     0,            0, (2-phi)*L**2],
                        [-12,    0,            0,         -6*L,   12,     0,            0,         -6*L],
                        [0,    -12,          6*L,            0,    0,    12,          6*L,            0],
                        [0,   -6*L, (2-phi)*L**2,            0,    0,   6*L, (4+phi)*L**2,            0],
                        [6*L,    0,            0, (2-phi)*L**2, -6*L,     0,            0, (4+phi)*L**2]])

        K = self.E * self.Ie * K/((1 + phi)*L**3)

        return K

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.

        Returns
        -------
        Gyroscopic matrix for the shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(1, 0.25, 0, 0.05, 211e9, 81.2e9, 7810,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[-0.        , -0.01943344, -0.00022681, -0.        ],
               [-0.01943344, -0.        , -0.        , -0.00022681],
               [ 0.00022681, -0.        , -0.        ,  0.0001524 ],
               [-0.        ,  0.00022681, -0.0001524 , -0.        ]])

        """
        phi = self.phi
        L = self.L

        G = np.zeros((8, 8))

        if self.gyroscopic:
            g1 = 36
            g2 = (3 - 15 * phi) * L
            g3 = (4 + 5 * phi + 10 * phi**2) * L**2
            g4 = (-1 - 5 * phi + 5 * phi**2) * L**2

            G = np.array([[  0,  g1,  g2,   0,   0,  g1,  g2,   0],
                          [ g1,   0,   0,  g2, -g1,   0,   0,  g2],
                          [-g2,   0,   0, -g3,  g2,   0,   0, -g4],
                          [  0, -g2,  g3,   0,   0,  g2,  g4,   0],
                          [  0,  g1, -g2,   0,   0, -g1, -g2,   0],
                          [-g1,   0,   0, -g2,  g1,   0,   0, -g2],
                          [-g2,   0,   0, -g4,  g2,   0,   0, -g3],
                          [  0, -g2,  g4,   0,   0,  g2,  g3,   0]])

            G = - self.rho * self.Ie * G / (15 * L * (1 + phi)**2)

        return G

        #  TODO stiffness Matrix due to an axial load
        #  TODO stiffness Matrix due to an axial torque
        #  TODO add speed as an argument so that skew-symmetric stiffness matrix can be evaluated (default to None)
        #  TODO skew-symmetric speed dependent contribution to element stiffness matrix from the internal damping
        #  TODO add tappered element. Modify shaft element to accept i_d and o_d as a list with to entries.


class DiskElement(object):
    #  TODO detail this class attributes inside the docstring
    """A disk element.

    This class will create a disk element.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    rho: float
        Mass density.
    width: float
        The disk width.
    i_d: float
        Inner diameter.
    o_d: float
        Outer diameter.

    Attributes
    ----------
    m : float
        Mass of the disk element.
    Id : float
        Diametral moment of inertia.
    Ip : float
        Polar moment of inertia

    References
    ----------
    .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
       & AW Lees, published by Cambridge University Press, 2010 pp. 156-157.

    Examples
    --------
    >>> disk = DiskElement(0, 7810, 0.07, 0.05, 0.28)
    >>> disk.Ip
    0.32956362089137037
    """

    #  TODO add __repr__ to the class
    def __init__(self, n, rho, width, i_d, o_d):
        self.n = n
        self.rho = rho
        self.width = width
        self.i_d = i_d
        self.o_d = o_d
        self.m = 0.25 * rho * np.pi * width * (o_d**2 - i_d**2)
        self.Id = 0.015625 * rho * np.pi * width*(o_d**4 - i_d**4) + self.m*(width**2)/12
        self.Ip = 0.03125 * rho * np.pi * width * (o_d**4 - i_d**4)

    def M(self):
        """
        This method will return the mass matrix for an instance of a disk
        element.

        Parameters
        ----------
        self

        Returns
        -------
        Mass matrix for the disk element.

        Examples
        --------
        >>> disk = DiskElement(0, 7810, 0.07, 0.05, 0.28)
        >>> disk.M()
        array([[ 32.58972765,   0.        ,   0.        ,   0.        ],
               [  0.        ,  32.58972765,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.17808928,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.17808928]])
        """
        m = self.m
        Id = self.Id

        M = np.array([[m, 0,  0,  0],
                       [0, m,  0,  0],
                       [0, 0, Id,  0],
                       [0, 0,  0, Id]])

        return M

    def G(self):
        """
        This method will return the gyroscopic matrix for an instance of a disk
        element.

        Parameters
        ----------
        self

        Returns
        -------
        Gyroscopic matrix for the disk element.

        Examples
        --------
        >>> disk = DiskElement(0, 7810, 0.07, 0.05, 0.28)
        >>> disk.G()
        array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.32956362],
               [ 0.        ,  0.        , -0.32956362,  0.        ]])
        """

        Ip = self.Ip

        G = np.array([[0, 0,   0,  0],
                       [0, 0,   0,  0],
                       [0, 0,   0, Ip],
                       [0, 0, -Ip,  0]])

        return G


class BearingElement(object):
    #  TODO detail this class attributes inside the docstring
    """A bearing element.

    This class will create a bearing element.
    Parameters can be a constant value or speed dependent.
    For speed dependent parameters, each argument should be passed
    as an array and the correspondent speed values should also be
    passed as an array.
    Values for each parameter will be interpolated for the speed.

    Parameters
    ----------
    kxx: float, array
        Direct stiffness in the x direction.
    cxx: float, array
        Direct damping in the x direction.
    kyy: float, array, optional
        Direct stiffness in the y direction.
        (defaults to kxx)
    cyy: float, array, optional
        Direct damping in the y direction.
        (defaults to cxx)
    kxy: float, array, optional
        Cross coupled stiffness in the x direction.
        (defaults to 0)
    cxy: float, array, optional
        Cross coupled damping in the x direction.
        (defaults to 0)
    kyx: float, array, optional
        Cross coupled stiffness in the y direction.
        (defaults to 0)
    cyx: float, array, optional
        Cross coupled damping in the y direction.
        (defaults to 0)
    w: array, optional
        Array with the speeds (rad/s).

    Examples
    --------

    """

    #  TODO implement for more complex cases (kxy, kthetatheta etc.)
    #  TODO consider kxx, kxy, kyx, kyy, cxx, cxy, cyx, cyy, mxx, myy, myx, myy (to import from XLTRC)
    #  TODO add speed as an argument
    #  TODO arguments should be lists related to speed
    #  TODO evaluate the use of pandas tables to display
    #  TODO create tests to evaluate interpolation
    #  TODO create tests for different cases of bearing instantiation
    def __init__(self, n,
                 kxx, cxx,
                 kyy=None, kxy=0, kyx=0,
                 cyy=None, cxy=0, cyx=0,
                 w=None):

        if w is not None:
            for arg in permutations([kxx, cxx, w], 2):
                if arg[0].shape != arg[1].shape:
                    raise Exception('kxx, cxx and w must have the same dimension')

        # set values for speed so that interpolation can be created
        if w is None:
            w = np.linspace(0, 10000, 4)

        if kyy is None:
            kyy = kxx
        if cyy is None:
            cyy = cxx
        # adjust array size to avoid error in interpolation
        if isinstance(kxy, (int, float)):
            kxy = [kxy for i in range(len(w))]
        if isinstance(kyx, (int, float)):
            kyx = [kyx for i in range(len(w))]
        if isinstance(cxy, (int, float)):
            cxy = [cxy for i in range(len(w))]
        if isinstance(cyx, (int, float)):
            cyx = [cyx for i in range(len(w))]

        args = {'kxx': kxx, 'cxx': cxx,
                'kyy': kyy, 'kxy': kxy, 'kyx': kyx,
                'cyy': cyy, 'cxy': cxy, 'cyx': cyx}

        self.n = n
        self.w = w

        for arg, val in args.items():
            if isinstance(val, (int, float)):
                # set values for each val so that interpolation can be created
                val = [val for i in range(4)]
            interp_func = interpolate.UnivariateSpline(w, val)
            setattr(self, arg, interp_func)

    def __repr__(self):
        return '%s' % self.__class__.__name__

    def K(self, w):
        kxx = self.kxx(w)
        kyy = self.kyy(w)
        kxy = self.kxy(w)
        kyx = self.kyx(w)

        K = np.array([[kxx, kxy],
                      [kyx, kyy]])

        return K

    def C(self, w):
        cxx = self.cxx(w)
        cyy = self.cyy(w)
        cxy = self.cxy(w)
        cyx = self.cyx(w)

        C = np.array([[cxx, cxy],
                      [cyx, cyy]])

        return C


class SealElement(BearingElement):
    pass
