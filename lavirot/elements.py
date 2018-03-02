import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from itertools import permutations
from collections import Iterable
from lavirot.materials import Material


__all__ = ["ShaftElement", "LumpedDiskElement", "DiskElement",
           "BearingElement", "SealElement"]


c_pal = {'red': '#C93C3C',
         'blue': '#0760BA',
         'green': '#2ECC71',
         'dark blue': '#07325E',
         'purple': '#A349C6',
         'grey': '#2D2D2D',
         'green2': '#08A4AF'}


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
    material : lavirot.material
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
    >>> from lavirot.materials import steel
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

            G = np.array([[  0, -g1,  g2,   0,   0,  g1,  g2,   0],
                          [ g1,   0,   0,  g2, -g1,   0,   0,  g2],
                          [-g2,   0,   0, -g3,  g2,   0,   0, -g4],
                          [  0, -g2,  g3,   0,   0,  g2,  g4,   0],
                          [  0,  g1, -g2,   0,   0, -g1, -g2,   0],
                          [-g1,   0,   0, -g2,  g1,   0,   0, -g2],
                          [-g2,   0,   0, -g4,  g2,   0,   0, -g3],
                          [  0, -g2,  g4,   0,   0,  g2,  g3,   0]])

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
        material : lavirot.material
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
        >>> from lavirot.materials import steel
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
        """Load shaft from xltrc.

        This method will construct a shaft loading the geometry
        and materials from a xltrc file.

        Parameters
        ----------
        file : str
            File path name.
        sheet_name : str
            Shaft sheet name. Default is 'Model'.

        Returns
        -------
        shaft : list
            List with the shaft elements.

        Examples
        --------
        """
        df = pd.read_excel(file, sheet_name=sheet_name)

        geometry = pd.DataFrame(df.iloc[19:])
        geometry = geometry.rename(columns=df.loc[18])
        geometry = geometry.dropna(thresh=3)

        material = df.iloc[3:13, 9:15]
        material = material.rename(columns=df.iloc[0])
        material = material.dropna(axis=0, how='all')

        # change to SI units
        if df.iloc[1, 1] == 'inches':
            for dim in ['length', 'od_Left', 'id_Left',
                        'od_Right', 'id_Right']:
                geometry[dim] = geometry[dim] * 0.0254

            geometry['axial'] = geometry['axial'] * 4.44822161

            for prop in ['Elastic Modulus E', 'Shear Modulus G']:
                material[prop] = material[prop] * 6894.757

            material['Density   r'] = material['Density   r'] * 27679.904

        colors = ['#636363', '#969696', '#bdbdbd', '#d9d9d9',
                  '#636363', '#969696', '#bdbdbd', '#d9d9d9']
        materials = {}
        for i, mat in material.iterrows():
            materials[mat.Material] = Material(
                name=f'Material{mat["Material"]}',
                rho=mat['Density   r'],
                E=mat['Elastic Modulus E'],
                G_s=mat['Shear Modulus G'],
                color=colors[mat['Material']]
            )

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
        >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
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
        """Load lumped masses from xltrc.

        This method will construct a disks list with loaded data
        from a xltrc file.

        Parameters
        ----------
        file : str
            File path name.
        sheet_name : str
            Masses sheet name. Default is 'More'.

        Returns
        -------
        disks : list
            List with the shaft elements.

        Examples
        --------
        """
        df = pd.read_excel(file, sheet_name=sheet_name)

        df_masses = pd.DataFrame(df.iloc[4:, :4])
        df_masses = df_masses.rename(columns=df.iloc[1, 1:4])
        df_masses = df_masses.rename(columns={' Added Mass & Inertia': 'n'})

        # convert to SI units
        if df.iloc[2, 1] == 'lbm':
            df_masses['Mass'] = df_masses['Mass'] * 0.45359237
            df_masses['Ip'] = df_masses['Ip'] * 0.00029263965342920005
            df_masses['It'] = df_masses['It'] * 0.00029263965342920005

        disks = [cls(d.n-1, d.Mass, d.It, d.Ip)
                 for _, d in df_masses.iterrows()]

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
    >>> from lavirot.materials import steel
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

        # TODO try to simplify the rest of the function putting
        # TODO everything in this for loop
        args = locals()
        for k, v in args.items():
            if isinstance(v, Iterable):
                args[k] = np.array(v, dtype=np.float64)

        kxx = args['kxx']
        cxx = args['cxx']
        kyy = args['kyy']
        kxy = args['kxy']
        kyx = args['kyx']
        cyy = args['cyy']
        cxy = args['cxy']
        cyx = args['cyx']
        w = args['w']

        # check for args consistency
        if w is not None:
            for arg in permutations([kxx, cxx, w], 2):
                if arg[0].shape != arg[1].shape:
                    raise Exception(
                        'kxx, cxx and w must have the same dimension'
                    )

        # set values for speed so that interpolation can be created
        if w is None:
            for arg in [kxx, cxx,
                        kyy, kxy, kyx,
                        cyy, cxy, cyx]:
                if isinstance(arg, np.ndarray):
                    raise Exception(
                        'w should be an array with the parameters dimension'
                    )
            w = np.linspace(0, 10000, 4)

        w = np.array(w, dtype=np.float)

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
        self.n_l = n
        self.n_r = n

        self.w = w
        self.color = '#355d7a'

        for arg, val in args.items():
            if isinstance(val, (int, float)):
                # set values for each val so that interpolation can be created
                val = [val for i in range(4)]
            interp_func = interpolate.UnivariateSpline(w, val)
            setattr(self, 'interpolated_' + arg, interp_func)

    def __repr__(self):
        return '%s' % self.__class__.__name__

    def K(self, w):
        kxx = self.interpolated_kxx(w)
        kyy = self.interpolated_kyy(w)
        kxy = self.interpolated_kxy(w)
        kyx = self.interpolated_kyx(w)

        K = np.array([[kxx, kxy],
                      [kyx, kyy]])

        return K

    def C(self, w):
        cxx = self.interpolated_cxx(w)
        cyy = self.interpolated_cyy(w)
        cxy = self.interpolated_cxy(w)
        cyx = self.interpolated_cyx(w)

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

    def plot_k_curve(self, w=None, ax=None,
                     kxx=True, kxy=True, kyx=True, kyy=True, **kwargs):
        """Plot the k curve fit.

        This method will plot the curve fit for the
        given speed range.

        Parameters
        ----------
        w : array, optional
            Speeds for which the plot will be made.
            If not provided, will use speed from bearing creation.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        kxx : bool, optional
            Whether or not kxx is plotted. Default is True.
        kxy : bool, optional
            Whether or not kxy is plotted. Default is True.
        kyx : bool, optional
            Whether or not kyx is plotted. Default is True.
        kyy : bool, optional
            Whether or not kyy is plotted. Default is True.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples
        --------
        """
        if w is None:
            w = self.w

        if ax is None:
            ax = plt.gca()

        if kxx is True:
            ax.plot(w, self.kxx(w), label='Kxx', **kwargs)
        if kyy is True:
            ax.plot(w, self.kyy(w), label='Kyy', **kwargs)
        if kxy is True:
            ax.plot(w, self.kxy(w), '--', label='Kxy', **kwargs)
        if kyx is True:
            ax.plot(w, self.kyx(w), '--', label='Kyx', **kwargs)

        ax.set_xlabel(r'Speed (rad/s)')
        ax.set_ylabel(r'Stiffness ($N/m$)')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.legend()

        return ax

    def plot_c_curve(self, w=None, ax=None,
                     cxx=True, cxy=True, cyx=True, cyy=True, **kwargs):
        """Plot the k curve fit.

        This method will plot the curve fit for the
        given speed range.

        Parameters
        ----------
        w : array, optional
            Speeds for which the plot will be made.
            If not provided, will use speed from bearing creation.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        cxx : bool, optional
            Whether or not cxx is plotted. Default is True.
        cxy : bool, optional
            Whether or not cxy is plotted. Default is True.
        cyx : bool, optional
            Whether or not cyx is plotted. Default is True.
        cyy : bool, optional
            Whether or not cyy is plotted. Default is True.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples
        --------
        """
        if w is None:
            w = self.w

        if ax is None:
            ax = plt.gca()
        if cxx is True:
            ax.plot(w, self.cxx(w), label='Cxx', **kwargs)
        if cyy is True:
            ax.plot(w, self.cyy(w), label='Cyy', **kwargs)
        if cxy is True:
            ax.plot(w, self.cxy(w), '--', label='Cxy', **kwargs)
        if cyx is True:
            ax.plot(w, self.cyx(w), '--', label='Cyx', **kwargs)

        ax.set_xlabel(r'Speed (rad/s)')
        ax.set_ylabel(r'Damping ($Ns/m$)')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.legend()

        return ax

    @classmethod
    def load_from_xltrc(cls, n, file, sheet_name='XLUseKCM'):
        """Load bearing from xltrc.

        This method will construct a bearing loading the coefficients
        from an xltrc file.

        Parameters
        ----------
        n: int
            Node in which the bearing will be inserted.
        file : str
            File path name.
        sheet_name : str
            Bearing sheet name. Default is 'XLUseKCM'.

        Returns
        -------
        bearing : lr.BearingElement
            A bearing element.

        Examples
        --------
        """
        # TODO Check .xls units to see if argument provided is consistent

        df = pd.read_excel(file, sheet_name=sheet_name)

        start_col = 0

        if sheet_name == 'XLTFPBrg':
            start_col = 2

        # locate were table starts
        for i, row in df.iterrows():
            if row.iloc[start_col] == 'Speed':
                start = i + 2

        df_bearing = pd.DataFrame(df.iloc[start:, start_col:])
        if sheet_name == 'XLTFPBrg':
            df_bearing = df_bearing.iloc[:10]
        df_bearing = df_bearing.rename(columns=df.loc[start - 2])
        df_bearing = df_bearing.dropna(axis=0, thresh=5)
        df_bearing = df_bearing.dropna(axis=1, thresh=5)

        if df.iloc[start - 1, 1] == 'lb/in':
            for col in df_bearing.columns:
                if col != 'Speed':
                    df_bearing[col] = df_bearing[col] * 175.126835

        df_bearing['Speed'] = df_bearing['Speed'] * 2 * np.pi / 60

        w = df_bearing.Speed.values
        kxx = df_bearing.Kxx.values
        kxy = df_bearing.Kxy.values
        kyx = df_bearing.Kyx.values
        kyy = df_bearing.Kyy.values
        cxx = df_bearing.Cxx.values
        cxy = df_bearing.Cxy.values
        cyx = df_bearing.Cyx.values
        cyy = df_bearing.Cyy.values

        return cls(n=n, w=w, kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                   cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy)


class SealElement(BearingElement):
    def __init__(self, n,
                 kxx, cxx,
                 kyy=None, kxy=0, kyx=0,
                 cyy=None, cxy=0, cyx=0,
                 w=None):
        super().__init__(n=n, w=w,
                         kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                         cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy)

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
