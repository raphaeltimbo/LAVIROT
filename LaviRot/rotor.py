import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las
from LaviRot.elements import *


__all__ = ['Rotor', 'rotor_example']


class Rotor(object):
    r"""A rotor object.

    This class will create a rotor with the shaft,
    disk and bearing elements provided.

    Parameters
    ----------
    shaft_elements: list
        List with the shaft elements
    disk_elements: list
        List with the disk elements
    bearing_elements: list
        List with the bearing elements
    w: float, optional
        Rotor speed. Defaults to 0.

    Returns
    -------
    A rotor object.

    Attributes
    ----------
    evalues : array
        Rotor's eigenvalues.
    evectors : array
        Rotor's eigenvectors.
    wn : array
        Rotor's natural frequencies in rad/s.
    wd : array
        Rotor's damped natural frequencies in rad/s.

    Examples
    --------
    >>> #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
    >>> n = 1
    >>> z = 0
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> E = 211e9
    >>> G = 81.2e9
    >>> rho = 7810
    >>> tim0 = ShaftElement(0, le, i_d, o_d, E, G, rho,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> tim1 = ShaftElement(1, le, i_d, o_d, E, G, rho,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> shaft_elm = [tim0, tim1]
    >>> disk0 = DiskElement(1, rho, 0.07, 0.05, 0.28)
    >>> stf = 1e6
    >>> bearing0 = BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = BearingElement(2, kxx=stf, cxx=0)
    >>> rotor = Rotor(shaft_elm, [disk0], [bearing0, bearing1])
    >>> rotor.wd[0] # doctest: +ELLIPSIS
    215.3707...
    """

    def __init__(self, shaft_elements, disk_elements, bearing_seal_elements, w=0):
        #  TODO consider speed as a rotor property. Setter should call __init__ again
        self._w = w
        self.shaft_elements = shaft_elements
        self.bearing_seal_elements = bearing_seal_elements
        self.disk_elements = disk_elements
        # Values for evalues and evectors will be calculated by self._calc_system
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        #  TODO check when disk diameter in no consistent with shaft diameter
        #  TODO add error for elements added to the same n (node)
        # number of dofs
        self.ndof = 4 * len(shaft_elements) + 4

        #  nodes axial position
        nodes_pos = [0]
        length = 0
        for sh in shaft_elements:
            length += sh.L
            nodes_pos.append(length)
        self.nodes_pos = nodes_pos
        self.nodes = [i for i in range(len(self.nodes_pos))]

        #  TODO for tappered elements i_d and o_d will be a list with two elements
        #  diameter at node position
        nodes_i_d = [s.i_d for s in self.shaft_elements]
        # append i_d for last node
        nodes_i_d.append(self.shaft_elements[-1].i_d)
        self.nodes_i_d = nodes_i_d

        nodes_o_d = [s.o_d for s in self.shaft_elements]
        # append o_d for last node
        nodes_o_d.append(self.shaft_elements[-1].o_d)
        self.nodes_o_d = nodes_o_d

        # call self._calc_system() to calculate current evalues and evectors
        self._calc_system()

    def _calc_system(self):
        self.evalues, self.evectors = self._eigen(self._w)
        self.wn = (np.absolute(self.evalues))[:self.ndof//2]
        self.wd = (np.imag(self.evalues))[:self.ndof//2]

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value
        self._calc_system()

    @staticmethod
    def _dofs(element):
        """The first and last dof for a given element"""
        if type(element) == ShaftElement:
            node = element.n
            n1 = 4 * node
            n2 = n1 + 8
        if type(element) == DiskElement:
            node = element.n
            n1 = 4 * node
            n2 = n1 + 4
        if isinstance(element, BearingElement):
            node = element.n
            n1 = 4 * node
            n2 = n1 + 2
        # TODO implement this for bearing with more dofs
        return n1, n2

    def M(self):
        r"""Mass matrix for an instance of a rotor.

        Returns
        -------
        Mass matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        #  Create the matrices
        M0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        for elm in self.disk_elements:
            n1, n2 = self._dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        return M0

    def K(self, w=None):
        """Stiffness matrix for an instance of a rotor.

        Returns
        -------
        Stiffness matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.K()[:4, :4]/1e6)
        array([[ 47.,   0.,   0.,   6.],
               [  0.,  46.,  -6.,   0.],
               [  0.,  -6.,   1.,   0.],
               [  6.,   0.,   0.,   1.]])
        """
        if w is None:
            w = self.w
        #  Create the matrices
        K0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            K0[n1:n2, n1:n2] += elm.K()

        for elm in self.bearing_seal_elements:
            n1, n2 = self._dofs(elm)
            K0[n1:n2, n1:n2] += elm.K(w)
        #  Skew-symmetric speed dependent contribution to element stiffness matrix
        #  from the internal damping.
        #  TODO add the contribution for K1 matrix

        return K0

    def C(self, w=None):
        """Damping matrix for an instance of a rotor.

        Returns
        -------
        Damping matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.C()[:4, :4]
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
        """
        if w is None:
            w = self.w
        #  Create the matrices
        C0 = np.zeros((self.ndof, self.ndof))

        for elm in self.bearing_seal_elements:
            n1, n2 = self._dofs(elm)
            C0[n1:n2, n1:n2] += elm.C(w)

        return C0

    def G(self):
        """Gyroscopic matrix for an instance of a rotor.

        Returns
        -------
        Gyroscopic matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.G()[:4, :4]
        array([[ 0.        , -0.01943344, -0.00022681,  0.        ],
               [-0.01943344,  0.        ,  0.        , -0.00022681],
               [ 0.00022681,  0.        ,  0.        ,  0.0001524 ],
               [ 0.        ,  0.00022681, -0.0001524 ,  0.        ]])
        """
        #  Create the matrices
        G0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        for elm in self.disk_elements:
            n1, n2 = self._dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        return G0

    def A(self, w=None):
        """State space matrix for an instance of a rotor.

        Returns
        -------
        State space matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.A()[50:56, :2])
        array([[     0.,  11110.],
               [-11106.,     -0.],
               [  -169.,      0.],
               [    -0.,   -169.],
               [    -0.,  10511.],
               [-10507.,     -0.]])
        """
        if w is None:
            w = self.w

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)
        #  TODO implement K(w) and C(w) for shaft, bearings etc.
        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M(), self.K(w)), la.solve(-self.M(), (self.C(w) + self.G()*w))])])

        return A

    @staticmethod
    def _index(eigenvalues):
        r"""Function used to generate an index that will sort
        eigenvalues and eigenvectors based on the imaginary (wd)
        part of the eigenvalues. Positive eigenvalues will be
        positioned at the first half of the array.

        Parameters
        ----------
        eigenvalues: array
            Array with the eigenvalues.

        Returns
        -------
        idx:
            An array with indices that will sort the
            eigenvalues and eigenvectors.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0, sorted_=False)
        >>> idx = rotor._index(evalues)
        >>> idx[:6] # doctest: +ELLIPSIS
        array([ 1,  3,  5,  7,  9, 11]...
        """
        # avoid float point errors when sorting
        evals_truncated = np.around(eigenvalues, decimals=5)
        a = np.imag(evals_truncated)  # First column
        b = np.absolute(evals_truncated)  # Second column
        ind = np.lexsort((b, a))  # Sort by imag, then by absolute
        # Positive eigenvalues first
        positive = [i for i in ind[len(a) // 2:]]
        negative = [i for i in ind[:len(a) // 2]]

        idx = np.array([positive, negative]).flatten()

        return idx

    def _eigen(self, w=None, sorted_=True):
        r"""This method will return the eigenvalues and eigenvectors of the
        state space matrix A, sorted by the index method which considers
        the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False

        Parameters
        ----------
        w: float
            Rotor speed.

        Returns
        -------
        evalues: array
            An array with the eigenvalues
        evectors array
            An array with the eigenvectors

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0)
        >>> evalues[0].imag # doctest: +ELLIPSIS
        82.653...
        """
        if w is None:
            w = self.w

        evalues, evectors = las.eigs(self.A(w), k=12, sigma=0, ncv=24, which='LM')
        if sorted_ is False:
            return evalues, evectors

        idx = self._index(evalues)

        return evalues[idx], evectors[:, idx]

    def H(self, node, w, return_T=False):
        r"""Calculates the H matrix for a given node and natural frequency.

        The matrix H contains information about the whirl direction,
        the orbit minor and major axis and the orbit inclination.
        The matrix is calculated by :math:`H = T.T^T` where the
        matrix T is constructed using the eigenvector corresponding
        to the natural frequency of interest:

        .. math::
           :nowrap:

           \begin{eqnarray}
              \begin{bmatrix}
              u(t)\\
              v(t)
              \end{bmatrix}
              = \mathfrak{R}\Bigg(
              \begin{bmatrix}
              r_u e^{j\eta_u}\\
              r_v e^{j\eta_v}
              \end{bmatrix}\Bigg)
              e^{j\omega_i t}
              =
              \begin{bmatrix}
              r_u cos(\eta_u + \omega_i t)\\
              r_v cos(\eta_v + \omega_i t)
              \end{bmatrix}
              = {\bf T}
              \begin{bmatrix}
              cos(\omega_i t)\\
              sin(\omega_i t)
              \end{bmatrix}
           \end{eqnarray}

        Where :math:`r_u e^{j\eta_u}` e :math:`r_v e^{j\eta_v}` are the
        elements of the *i*\th eigenvector, corresponding to the node and
        natural frequency of interest (mode).

        .. math::

            {\bf T} =
            \begin{bmatrix}
            r_u cos(\eta_u) & -r_u sin(\eta_u)\\
            r_u cos(\eta_u) & -r_v sin(\eta_v)
            \end{bmatrix}

        Parameters
        ----------
        node: int
            Node for which the matrix H will be calculated.
        w: int
            Index corresponding to the natural frequency
            of interest.
        return_T: bool, optional
            If True, returns the H matrix and a dictionary with the
            values for :math:`r_u, r_v, \eta_u, \eta_v`.

            Default is false.

        Returns
        -------
        H: array
            Matrix H.
        Tdic: dict
            Dictionary with values for :math:`r_u, r_v, \eta_u, \eta_v`.

            It will be returned only if return_T is True.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # H matrix for the 0th node
        >>> rotor.H(0, 0) # doctest: +ELLIPSIS
        array([[  2.46154806e-29,  -7.17353298e-18],
               [ -7.17353298e-18,   2.11429917e-06]])


        """
        # get vector of interest based on freqs
        vector = self.evectors[4 * node:4 * node + 2, w]
        # get translation sdofs for specified node for each mode
        u = vector[0]
        v = vector[1]
        ru = np.absolute(u)
        rv = np.absolute(v)

        nu = np.angle(u)
        nv = np.angle(v)
        T = np.array([[ru * np.cos(nu), -ru * np.sin(nu)],
                      [rv * np.cos(nv), -rv * np.sin(nv)]])
        H = T @ T.T

        if return_T:
            Tdic = {'ru': ru,
                    'rv': rv,
                    'nu': nu,
                    'nv': nv}
            return H, Tdic

        return H

    def kappa(self, node, w, wd=True):
        r"""Calculates kappa for a given node and natural frequency.

        w is the the index of the natural frequency of interest.
        The function calculates the orbit parameter :math:`\kappa`:

        .. math::

            \kappa = \pm \sqrt{\lambda_2 / \lambda_1}

        Where :math:`\sqrt{\lambda_1}` is the length of the semiminor axes
        and :math:`\sqrt{\lambda_2}` is the length of the semimajor axes.

        If :math:`\kappa = \pm 1`, the orbit is circular.

        If :math:`\kappa` is positive we have a forward rotating orbit
        and if it is negative we have a backward rotating orbit.

        Parameters
        ----------
        node: int
            Node for which kappa will be calculated.
        w: int
            Index corresponding to the natural frequency
            of interest.
        wd: bool
            If True, damping natural frequencies are used.

            Default is true.

        Returns
        -------
        kappa: dict
            A dictionary with values for the natural frequency,
            major axis, minor axis and kappa.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # kappa for each node of the first natural frequency
        >>> # Major axes for node 0 and natural frequency (mode) 0.
        >>> rotor.kappa(0, 0)['Major axes'] # doctest: +ELLIPSIS
        array(0.00145...)
        >>> # kappa for node 2 and natural frequency (mode) 3.
        >>> rotor.kappa(2, 3)['kappa'] # doctest: +ELLIPSIS
        array(8.7006...e-13)

        """
        if wd:
            nat_freq = self.wd[w]
        else:
            nat_freq = self.wn[w]

        H, Tvals = self.H(node, w, return_T=True)
        nu = Tvals['nu']
        nv = Tvals['nv']

        lam = la.eig(H)[0]

        #  TODO normalize the orbit (after all orbits have been calculated?)
        # lam is the eigenvalue -> sqrt(lam) is the minor/major axis.
        # kappa encodes the relation between the axis and the precession.
        minor = np.sqrt(lam.min())
        major = np.sqrt(lam.max())
        kappa = minor / major
        diff = nv - nu

        # we need to evaluate if 0 < nv - nu < pi.
        if diff < -np.pi:
            diff += 2 * np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi

        # if nv = nu or nv = nu + pi then the response is a straight line.
        if diff == 0 or diff == np.pi:
            kappa = 0

        # if 0 < nv - nu < pi, then a backward rotating mode exists.
        elif 0 < diff < np.pi:
            kappa *= -1

        k = ({'Frequency': nat_freq,
              'Minor axes': np.real(minor),
              'Major axes': np.real(major),
              'kappa': np.real(kappa)})

        return k

    def kappa_mode(self, w):
        r"""This function evaluates kappa given the index of
        the natural frequency of interest.
        Values of kappa are evaluated for each node of the
        corresponding frequency mode.

        Parameters
        ----------
        w: int
            Index corresponding to the natural frequency
            of interest.

        Returns
        -------
        kappa_mode: list
            A list with the value of kappa for each node related
            to the mode/natural frequency of interest.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # kappa for each node of the first natural frequency
        >>> rotor.kappa_mode(0) # doctest: +ELLIPSIS
        [array(0.0), array(0.0), array(0.0), array(0.0), array(0.0), array(0.0), array(0.0)]
        """
        kappa_mode = [self.kappa(node, w)['kappa'] for node in self.nodes]
        return kappa_mode


    def orbit(self):
        pass
    #  TODO make w a property. Make eigen an attribute.
    #  TODO when w is changed, eigen is calculated and is available to methods.
    #  TODO static methods as auxiliary functions


def rotor_example():
    r"""This function returns an instance of a simple rotor with
    two shaft elements, one disk and two simple bearings.
    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Parameters
    ----------

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = rotor_example()
    >>> np.round(rotor.wd[:4])
    array([  83.,   87.,  255.,  274.])
    """

    #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
    i_d = 0
    o_d = 0.05
    E = 211e9
    Gs = 81.2e9
    rho = 7810
    n = 6
    nelem = [x for x in range(n)]
    L = [0.25 for i in range(n)]

    shaft_elem = [ShaftElement(n, l, i_d, o_d, E, Gs, rho,
                               shear_effects=True,
                               rotary_inertia=True,
                               gyroscopic=True) for n, l in zip(nelem, L)]

    disk0 = DiskElement(2, rho, 0.07, 0.05, 0.28)
    disk1 = DiskElement(4, rho, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

