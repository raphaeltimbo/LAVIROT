import warnings
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ross.data_io.read_xl import (
    load_bearing_seals_from_yaml, load_bearing_seals_from_xltrc,
    load_disks_from_xltrc, load_shaft_from_xltrc)


__all__ = ["ShaftElement", "LumpedDiskElement", "DiskElement",
           "BearingElement", "SealElement", "IsotSealElement"]


class Element:
    """Element class."""
    def __init__(self):
        pass

    def summary(self):
        """A summary for the element.

        A pandas series with the element properties as variables.
        """
        attributes = self.__dict__
        attributes['type'] = self.__class__.__name__
        return pd.Series(attributes)


class ShaftElement(Element):
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
    L : float
        Element length.
    i_d : float
        Inner diameter of the element.
    o_d : float
        Outer diameter of the element.
    material : ross.material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
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
    Poisson : float
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
    >>> from ross.materials import steel
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> Euler_Bernoulli_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                        shear_effects=False, rotary_inertia=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> Timoshenko_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                   rotary_inertia=True,
    ...                                   shear_effects=True)
    >>> Timoshenko_Element.phi
    0.08795566502463055
    """
    #  TODO detail this class attributes inside the docstring
    #  TODO add __repr__ to the class
    #  TODO add load from .xls -> sheet More
    def __init__(self, L, i_d, o_d, material,
                 n=None,
                 axial_force=0, torque=0,
                 shear_effects=True,
                 rotary_inertia=True,
                 gyroscopic=True
                 ):

        self.shear_effects = shear_effects
        self.rotary_inertia = rotary_inertia
        self.gyroscopic = gyroscopic

        self._n = n
        self.n_l = n
        self.n_r = None
        if n is not None:
            self.n_r = n + 1

        self.L = float(L)

        # diameters
        self.i_d = float(i_d)
        self.o_d = float(o_d)
        self.i_d_l = float(i_d)
        self.o_d_l = float(o_d)
        self.i_d_r = float(i_d)
        self.o_d_r = float(o_d)

        self.material = material
        self.material_name = material.name
        self.E = material.E
        self.G_s = material.G_s
        self.Poisson = material.Poisson
        self.color = material.color
        self.rho = material.rho
        self.A = np.pi*(o_d**2 - i_d**2)/4
        self.volume = self.A * self.L
        self.m = self.rho * self.volume
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
            # kappa = 6*r12*((1+self.poisson)/
            #           ((r12*(7 + 12*self.poisson + 4*self.poisson**2) +
            #             4*r2*(5 + 6*self.poisson + 2*self.poisson**2))))
            #  kappa as per Cowper (1996)
            kappa = 6*r12*((1+self.Poisson) /
                           ((r12*(7 + 6*self.Poisson) +
                           r2*(20 + 12*self.Poisson))))
            phi = 12*self.E*self.Ie/(self.G_s*kappa*self.A*L**2)

        self.phi = phi

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'(L={self.L:{0}.{5}}, i_d={self.i_d:{0}.{5}}, ' \
               f'o_d={self.o_d:{0}.{5}}, material={self.material!r}, ' \
               f'n={self.n})'

    def __str__(self):
        return (
            f'\nElem. N:    {self.n}'
            f'\nLenght:     {self.L:{10}.{5}}'
            f'\nInt. Diam.: {self.i_d:{10}.{5}}'
            f'\nOut. Diam.: {self.o_d:{10}.{5}}'
            f'\n{35*"-"}'
            f'\n{self.material}'
            f'\n'
        )

    def M(self):
        r"""Mass matrix for an instance of a shaft element.

        Returns
        -------
        Mass matrix for the shaft element.

        Examples
        --------
        >>> from lavirot.materials import steel
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
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
        # fmt: off
        M = np.array([[m01,     0,     0,   m02,   m03,     0,     0,   m04],
                      [  0,   m01,  -m02,     0,     0,   m03,  -m04,     0],
                      [  0,  -m02,   m05,     0,     0,   m04,   m06,     0],
                      [m02,     0,     0,   m05,  -m04,     0,     0,   m06],
                      [m03,     0,     0,  -m04,   m01,     0,     0,  -m02],
                      [  0,   m03,   m04,     0,     0,   m01,   m02,     0],
                      [  0,  -m04,   m06,     0,     0,   m02,   m05,     0],
                      [m04,     0,     0,   m06,  -m02,     0,     0,   m05]])
        # fmt: on
        M = self.rho * self.A * self.L * M/(840*(1 + phi)**2)

        if self.rotary_inertia:
            ms1 = 36
            ms2 = (3 - 15*phi)*L
            ms3 = (4 + 5*phi + 10*phi**2)*L**2
            ms4 = (-1 - 5*phi + 5*phi**2)*L**2
            # fmt: off
            Ms = np.array([[ms1,      0,     0,   ms2,  -ms1,     0,     0,   ms2],
                           [   0,   ms1,  -ms2,     0,     0,  -ms1,  -ms2,     0],
                           [   0,  -ms2,   ms3,     0,     0,   ms2,   ms4,     0],
                           [ ms2,     0,     0,   ms3,  -ms2,     0,     0,   ms4],
                           [-ms1,     0,     0,  -ms2,   ms1,     0,     0,  -ms2],
                           [   0,  -ms1,   ms2,     0,     0,   ms1,   ms2,     0],
                           [   0,  -ms2,   ms4,     0,     0,   ms2,   ms3,     0],
                           [ ms2,     0,     0,   ms4,  -ms2,     0,     0,   ms3]])
            # fmt: on
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
        >>> from lavirot.materials import steel
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
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
        # fmt: off
        K = np.array([
            [12,     0,            0,          6*L,  -12,     0,            0,          6*L],
            [0,     12,         -6*L,            0,    0,   -12,         -6*L,            0],
            [0,   -6*L, (4+phi)*L**2,            0,    0,   6*L, (2-phi)*L**2,            0],
            [6*L,    0,            0, (4+phi)*L**2, -6*L,     0,            0, (2-phi)*L**2],
            [-12,    0,            0,         -6*L,   12,     0,            0,         -6*L],
            [0,    -12,          6*L,            0,    0,    12,          6*L,            0],
            [0,   -6*L, (2-phi)*L**2,            0,    0,   6*L, (4+phi)*L**2,            0],
            [6*L,    0,            0, (2-phi)*L**2, -6*L,     0,            0, (4+phi)*L**2]
        ])
        # fmt: on
        K = self.E * self.Ie * K/((1 + phi)*L**3)

        return K

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.

        Returns
        -------
        Gyroscopic matrix for the shaft element.

        Examples
        --------
        >>> from lavirot.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[-0.        ,  0.01943344, -0.00022681, -0.        ],
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
            # fmt: off
            G = np.array([[  0, -g1,  g2,   0,   0,  g1,  g2,   0],
                          [ g1,   0,   0,  g2, -g1,   0,   0,  g2],
                          [-g2,   0,   0, -g3,  g2,   0,   0, -g4],
                          [  0, -g2,  g3,   0,   0,  g2,  g4,   0],
                          [  0,  g1, -g2,   0,   0, -g1, -g2,   0],
                          [-g1,   0,   0, -g2,  g1,   0,   0, -g2],
                          [-g2,   0,   0, -g4,  g2,   0,   0, -g3],
                          [  0, -g2,  g4,   0,   0,  g2,  g3,   0]])
            # fmt: on
            G = - self.rho * self.Ie * G / (15 * L * (1 + phi)**2)

        return G

    def patch(self, ax, position):
        """Shaft element patch.

        Patch that will be used to draw the shaft element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        position_u = [position, self.i_d]  # upper
        position_l = [position, -self.o_d]  # lower
        width = self.L
        height = self.o_d - self.i_d

        #  plot the upper half of the shaft
        ax.add_patch(mpatches.Rectangle(position_u, width, height,
                                        linestyle='--', linewidth=0.5,
                                        ec='k', fc=self.color, alpha=0.8))
        #  plot the lower half of the shaft
        ax.add_patch(mpatches.Rectangle(position_l, width, height,
                                        linestyle='--', linewidth=0.5,
                                        ec='k', fc=self.color, alpha=0.8))


    @classmethod
    def section(cls, L, ne,
                si_d, so_d, material,
                n=None,
                shear_effects=True,
                rotary_inertia=True,
                gyroscopic=True
                ):
        """Shaft section constructor.

        This method will create a shaft section with length 'L'
        divided into 'ne' elements.

        Parameters
        ----------
        i_d : float
            Inner diameter of the section.
        o_d : float
            Outer diameter of the section.
        E : float
            Young's modulus.
        G_s : float
            Shear modulus.
        material : ross.material
            Shaft material.
        n : int, optional
            Element number (coincident with it's first node).
            If not given, it will be set when the rotor is assembled
            according to the element's position in the list supplied to
            the rotor constructor.
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
        elements: list
            List with the 'ne' shaft elements.

        Examples
        --------
        >>> # shaft material
        >>> from ross.materials import steel
        >>> # shaft inner and outer diameters
        >>> si_d = 0
        >>> so_d = 0.01585
        >>> sec = ShaftElement.section(247.65e-3, 4, 0, 15.8e-3, steel)
        >>> len(sec)
        4
        >>> sec[0].i_d
        0
        """

        le = L / ne

        elements = [cls(le, si_d, so_d, material,
                        n,
                        shear_effects,
                        rotary_inertia,
                        gyroscopic)
                    for _ in range(ne)]

        return elements

    @classmethod
    def load_from_xltrc(cls, file, sheet_name='Model'):
        # TODO docstrings should be here not in the io module
        geometry, materials = load_shaft_from_xltrc(file, sheet_name)
        shaft = [ShaftElement(
            el.length, el.id_Left, el.od_Left,
            materials[el.matnum], n=el.elemnum-1)
            for i, el in geometry.iterrows()]

        return shaft

        #  TODO stiffness Matrix due to an axial load
        #  TODO stiffness Matrix due to an axial torque
        #  TODO add speed as an argument so that skew-symmetric stiffness matrix can be evaluated (default to None)
        #  TODO skew-symmetric speed dependent contribution to element stiffness matrix from the internal damping
        #  TODO add tappered element. Modify shaft element to accept i_d and o_d as a list with to entries.


class LumpedDiskElement(Element):
    """A lumped disk element.

     This class will create a lumped disk element.

     Parameters
     ----------
     n: int
         Node in which the disk will be inserted.
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
     >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
     >>> disk.Ip
     0.32956362
     """
    def __init__(self, n, m, Id, Ip):
        self.n = n
        self.n_l = n
        self.n_r = n

        self.m = m
        self.Id = Id
        self.Ip = Ip
        self.color = '#bc625b'

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
        >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.M()
        array([[ 32.58972765,   0.        ,   0.        ,   0.        ],
               [  0.        ,  32.58972765,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.17808928,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.17808928]])
        """
        m = self.m
        Id = self.Id
        # fmt: off
        M = np.array([[m, 0,  0,  0],
                       [0, m,  0,  0],
                       [0, 0, Id,  0],
                       [0, 0,  0, Id]])
        # fmt: on
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
        >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.G()
        array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.32956362],
               [ 0.        ,  0.        , -0.32956362,  0.        ]])
        """

        Ip = self.Ip
        # fmt: off
        G = np.array([[0, 0,   0,  0],
                      [0, 0,   0,  0],
                      [0, 0,   0, Ip],
                      [0, 0, -Ip,  0]])
        # fmt: on
        return G

    def patch(self, ax, position):
        """Lumped Disk element patch.

        Patch that will be used to draw the disk element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        D = ypos * 1.5
        hw = 0.005

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [[zpos, ypos],  # upper
                         [zpos + hw, ypos + D],
                         [zpos - hw, ypos + D],
                         [zpos, ypos]]
        disk_points_l = [[zpos, -ypos],  # lower
                         [zpos + hw, -(ypos + D)],
                         [zpos - hw, -(ypos + D)],
                         [zpos, -ypos]]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))

        ax.add_patch(mpatches.Circle(xy=(zpos, ypos + D),
                                     radius=0.01, color=self.color))
        ax.add_patch(mpatches.Circle(xy=(zpos, -(ypos + D)),
                                     radius=0.01, color=self.color))

    @classmethod
    def load_from_xltrc(cls, file, sheet_name='More'):
        df = load_disks_from_xltrc(file, sheet_name)
        disks = [cls(d.n-1, d.Mass, d.It, d.Ip)
                 for _, d in df.iterrows()]

        return disks


class DiskElement(LumpedDiskElement):
    #  TODO detail this class attributes inside the docstring
    """A disk element.

    This class will create a disk element.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    material : lavirot.Material
         Shaft material.
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
    >>> from ross.materials import steel
    >>> disk = DiskElement(0, steel, 0.07, 0.05, 0.28)
    >>> disk.Ip
    0.32956362089137037
    """

    #  TODO add __repr__ to the class
    def __init__(self, n, material, width, i_d, o_d):
        if not isinstance(n, int):
            raise TypeError(f'n should be int, not {n.__class__.__name__}')
        self.n = n
        self.n_l = n
        self.n_r = n

        self.material = material
        self.rho = material.rho
        self.width = width

        # diameters
        self.i_d = i_d
        self.o_d = o_d
        self.i_d_l = i_d
        self.o_d_l = o_d
        self.i_d_r = i_d
        self.o_d_r = o_d

        self.m = 0.25 * self.rho * np.pi * width * (o_d**2 - i_d**2)
        self.Id = (0.015625 * self.rho * np.pi * width*(o_d**4 - i_d**4)
                   + self.m*(width**2)/12)
        self.Ip = 0.03125 * self.rho * np.pi * width * (o_d**4 - i_d**4)
        self.color = '#bc625b'

        super().__init__(self.n, self.m, self.Id, self.Ip)

    def patch(self, ax, position):
        """Disk element patch.

        Patch that will be used to draw the disk element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        if isinstance(position, tuple):
            position = position[0]
        zpos = position
        ypos = self.i_d
        D = self.o_d
        hw = self.width / 2  # half width

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [[zpos, ypos],  # upper
                         [zpos + hw, ypos + 0.1 * D],
                         [zpos + hw, ypos + 0.9 * D],
                         [zpos - hw, ypos + 0.9 * D],
                         [zpos - hw, ypos + 0.1 * D],
                         [zpos, ypos]]
        disk_points_l = [[zpos, -ypos],  # lower
                         [zpos + hw, -(ypos + 0.1 * D)],
                         [zpos + hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.1 * D)],
                         [zpos, -ypos]]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))


class _Coefficient:
    def __init__(self, coefficient, w=None, interpolated=None):
        if isinstance(coefficient, (int, float)):
            if w is not None:
                coefficient = [coefficient for _ in range(len(w))]
            else:
                coefficient = [coefficient]

        self.coefficient = coefficient
        self.w = w

        if len(self.coefficient) > 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.interpolated = interpolate.UnivariateSpline(
                        self.w, self.coefficient
                    )
            #  dfitpack.error is not exposed by scipy
            #  so a bare except is used
            except:
                raise ValueError('Arguments (coefficients and w)'
                                 ' must have the same dimension')
        else:
            self.interpolated = lambda x: np.array(self.coefficient[0])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        w_range = np.linspace(min(self.w), max(self.w), 30)

        ax.plot(w_range, self.interpolated(w_range), **kwargs)
        ax.set_xlabel('Speed (rad/s)')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        return ax


class _Stiffness_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel('Stiffness ($N/m$)')

        return ax


class _Damping_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel('Damping ($Ns/m$)')

        return ax


class BearingElement(Element):
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

        args = ['kxx', 'kyy', 'kxy', 'kyx',
                'cxx', 'cyy', 'cxy', 'cyx']

        # all args to coefficients
        args_dict = locals()
        coefficients = {}

        if kyy is None:
            args_dict['kyy'] = kxx
        if cyy is None:
            args_dict['cyy'] = cxx

        for arg in args:
            if arg[0] == 'k':
                coefficients[arg] = _Stiffness_Coefficient(
                    args_dict[arg], args_dict['w'])
            else:
                coefficients[arg] = _Damping_Coefficient(
                    args_dict[arg], args_dict['w'])

        coefficients_len = [len(v.coefficient) for v in coefficients.values()]

        if w is not None:
            coefficients_len.append(len(args_dict['w']))
            if len(set(coefficients_len)) > 1:
                raise ValueError('Arguments (coefficients and w)'
                                 ' must have the same dimension')
        else:
            for c in coefficients_len:
                if c != 1:
                    raise ValueError('Arguments (coefficients and w)'
                                     ' must have the same dimension')

        for k, v in coefficients.items():
            setattr(self, k, v)

        self.n = n
        self.n_l = n
        self.n_r = n

        self.w = np.array(w, dtype=np.float64)
        self.color = '#355d7a'

    def __repr__(self):
        return '%s' % self.__class__.__name__

    def K(self, w):
        kxx = self.kxx.interpolated(w)
        kyy = self.kyy.interpolated(w)
        kxy = self.kxy.interpolated(w)
        kyx = self.kyx.interpolated(w)

        K = np.array([[kxx, kxy],
                      [kyx, kyy]])

        return K

    def C(self, w):
        cxx = self.cxx.interpolated(w)
        cyy = self.cyy.interpolated(w)
        cxy = self.cxy.interpolated(w)
        cyx = self.cyx.interpolated(w)

        C = np.array([[cxx, cxy],
                      [cyx, cyy]])

        return C

    def patch(self, ax, position):
        """Bearing element patch.

        Patch that will be used to draw the bearing element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position (z, y) in which the patch will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        h = -0.75 * ypos  # height

        #  node (x pos), outer diam. (y pos)
        bearing_points = [[zpos, ypos],  # upper
                          [zpos + h / 2, ypos - h],
                          [zpos - h / 2, ypos - h],
                          [zpos, ypos]]
        ax.add_patch(mpatches.Polygon(bearing_points, color=self.color, picker=True))

    @classmethod
    def load_from_yaml(cls, n, file):
        kwargs = load_bearing_seals_from_yaml(file)
        return cls(n, **kwargs)

    @classmethod
    def load_from_xltrc(cls, n, file, sheet_name='XLUseKCM'):
        kwargs = load_bearing_seals_from_xltrc(file, sheet_name)
        return cls(n, **kwargs)


class SealElement(BearingElement):
    def __init__(self, n,
                 kxx, cxx,
                 kyy=None, kxy=0, kyx=0,
                 cyy=None, cxy=0, cyx=0,
                 w=None, seal_leakage=None):
        super().__init__(n=n, w=w,
                         kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                         cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy)

        self.seal_leakage = seal_leakage
        self.color = '#77ACA2'

    def patch(self, ax, position):
        """Seal element patch.

        Patch that will be used to draw the seal element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position in which the patch will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        hw = 0.05
        # TODO adapt hw according to bal drum diameter

        #  node (x pos), outer diam. (y pos)
        seal_points_u = [[zpos, ypos*1.1],  # upper
                         [zpos + hw, ypos*1.1],
                         [zpos + hw, ypos*1.3],
                         [zpos - hw, ypos*1.3],
                         [zpos - hw, ypos*1.1],
                         [zpos, ypos*1.1]]
        seal_points_l = [[zpos, -ypos*1.1],  # lower
                         [zpos + hw, -(ypos*1.1)],
                         [zpos + hw, -(ypos*1.3)],
                         [zpos - hw, -(ypos*1.3)],
                         [zpos - hw, -(ypos*1.1)],
                         [zpos, -ypos*1.1]]
        ax.add_patch(mpatches.Polygon(seal_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(seal_points_l, facecolor=self.color))


class IsotSealElement(SealElement):
    def __init__(self, n,
                 kxx, cxx,
                 kyy=None, kxy=0, kyx=0,
                 cyy=None, cxy=0, cyx=0,
                 w=None, seal_leakage=None,
                 absolute_viscosity=None, cell_vol_to_area_ratio=None, 
                 compressibility_factor=None, entrance_loss_coefficient=None,
                 exit_clearance=None, exit_recovery_factor=None, 
                 inlet_clearance=None, inlet_preswirl_ratio=None, 
                 molecular_weight=None, number_integr_steps=None, 
                 p_exit=None, p_supply=None,
                 reservoir_temperature=None, seal_diameter=None, seal_length=None,
                 specific_heat_ratio=None,
                 speed=None,
                 tolerance_percentage=None,
                 turbulence_coef_mr=None,
                 turbulence_coef_ms=None,
                 turbulence_coef_nr=None,
                 turbulence_coef_ns=None
                 ):
        super().__init__(n=n, w=w,
                         kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                         cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy,
                         seal_leakage=seal_leakage)

        self.absolute_viscosity = absolute_viscosity
        self.cell_vol_to_area_ratio = cell_vol_to_area_ratio
        self.compressibility_factor = compressibility_factor
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_clearance = exit_clearance
        self.exit_recovery_factor = exit_recovery_factor
        self.inlet_clearance = inlet_clearance
        self.inlet_preswirl_ratio = inlet_preswirl_ratio
        self.molecular_weight = molecular_weight
        self.number_integr_steps = number_integr_steps
        self.p_exit = p_exit
        self.p_supply = p_supply
        self.reservoir_temperature = reservoir_temperature
        self.seal_diameter = seal_diameter
        self.seal_length = seal_length
        self.specific_heat_ratio = specific_heat_ratio
        self.speed = speed
        self.tolerance_percentage = tolerance_percentage
        self.turbulence_coef_mr = turbulence_coef_mr
        self.turbulence_coef_ms = turbulence_coef_ms
        self.turbulence_coef_nr = turbulence_coef_nr
        self.turbulence_coef_ns = turbulence_coef_ns
