import os
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.signal as signal
import scipy.io as sio
from copy import copy
from collections import Iterable

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from LaviRot.elements import *
from LaviRot.materials import steel


__all__ = ['Rotor', 'rotor_example']

# set style and colors
plt.style.use('seaborn-white')
plt.style.use({
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 0.1,
    'grid.color': '.9',
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.2
    })

_orig_rc_params = mpl.rcParams.copy()

c_pal = {'red': '#C93C3C',
         'blue': '#0760BA',
         'green': '#2ECC71',
         'dark blue': '#07325E',
         'purple': '#A349C6',
         'grey': '#2D2D2D',
         'green2': '#08A4AF'}


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
    >>> from LaviRot.materials import steel
    >>> z = 0
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> tim0 = ShaftElement(le, i_d, o_d, steel,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> tim1 = ShaftElement(le, i_d, o_d, steel,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> shaft_elm = [tim0, tim1]
    >>> disk0 = DiskElement(1, steel, 0.07, 0.05, 0.28)
    >>> stf = 1e6
    >>> bearing0 = BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = BearingElement(2, kxx=stf, cxx=0)
    >>> rotor = Rotor(shaft_elm, [disk0], [bearing0, bearing1])
    >>> rotor.wd[0] # doctest: +ELLIPSIS
    215.3707...
    """

    def __init__(self, shaft_elements, disk_elements=None, bearing_seal_elements=None, w=0):
        #  TODO consider speed as a rotor property. Setter should call __init__ again
        self._w = w

        ####################################################
        # Config attributes
        ####################################################

        self.SPARSE = True

        ####################################################

        # flatten shaft_elements
        def flatten(l):
            for el in l:
                if isinstance(el, Iterable) \
                        and not isinstance(el, (str, bytes)):
                    yield from flatten(el)
                else:
                    yield el

        # flatten and make a copy for shaft elements to avoid altering
        # attributes for elements that might be used in different rotors
        # e.g. altering shaft_element.n
        shaft_elements = [copy(el) for el in flatten(shaft_elements)]

        # set n for each shaft element
        for i, sh in enumerate(shaft_elements):
            if sh.n is None:
                sh.n = i

        if disk_elements is None:
            disk_elements = []
        if bearing_seal_elements is None:
            bearing_seal_elements = []

        self.shaft_elements = shaft_elements
        self.bearing_seal_elements = bearing_seal_elements
        self.disk_elements = disk_elements

        # Values for evalues and evectors will be calculated by self._calc_system
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        self.H = None

        self._v0 = None  # used to call eigs
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
        self.evalues, self.evectors = self._eigen(self.w)
        wn_len = len(self.evalues) // 2
        self.wn = (np.absolute(self.evalues))[:wn_len]
        self.wd = (np.imag(self.evalues))[:wn_len]
        self.H = self._H()

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
        if isinstance(element, ShaftElement):
            node = element.n
            n1 = 4 * node
            n2 = n1 + 8
        if isinstance(element, LumpedDiskElement):
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
        array([[ 0.        ,  0.01943344, -0.00022681,  0.        ],
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
               [  -169.,     -0.],
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
        evals_truncated = np.around(eigenvalues, decimals=10)
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

        if self.SPARSE is True:
            try:
                evalues, evectors = las.eigs(self.A(w), k=12, sigma=0, ncv=24, which='LM', v0=self._v0)
                # store v0 as a linear combination of the previously
                # calculated eigenvectors to use in the next call to eigs
                self._v0 = np.real(sum(evectors.T))
            except las.ArpackError:
                evalues, evectors = la.eig(self.A(w))
        else:
            evalues, evectors = la.eig(self.A(w))

        if sorted_ is False:
            return evalues, evectors

        idx = self._index(evalues)

        return evalues[idx], evectors[:, idx]

    def H_kappa(self, node, w, return_T=False):
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
        >>> rotor.H_kappa(0, 0) # doctest: +ELLIPSIS
        array([[  8.78547006e-30,  -4.30647963e-18],
               [ -4.30647963e-18,   2.11429917e-06]])


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
        0.00145...
        >>> # kappa for node 2 and natural frequency (mode) 3.
        >>> rotor.kappa(2, 3)['kappa'] # doctest: +ELLIPSIS
        8.539...e-14
        """
        if wd:
            nat_freq = self.wd[w]
        else:
            nat_freq = self.wn[w]

        H, Tvals = self.H_kappa(node, w, return_T=True)
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
        [-0.0, -0.0, -0.0, -0.0, -1.153...e-08, -0.0, -1.239...e-08]
        """
        kappa_mode = [self.kappa(node, w)['kappa'] for node in self.nodes]
        return kappa_mode

    def orbit(self):
        pass
    #  TODO static methods as auxiliary functions

    def _H(self):
        r"""Continuous-time linear time invariant system.

        This method is used to create a Continuous-time linear
        time invariant system for the mdof system.
        From this system we can obtain poles, impulse response,
        generate a bode, etc.

        """
        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # x' = Ax + Bu
        B2 = I
        A = self.A()
        B = np.vstack([Z,
                       la.solve(self.M(), B2)])

        # y = Cx + Du
        # Observation matrices
        Cd = I
        Cv = Z
        Ca = Z

        # TODO Check equation below regarding gyroscopic matrix
        C = np.hstack((Cd - Ca @ la.solve(self.M(), self.K()), Cv - Ca @ la.solve(self.M(), self.C())))
        D = Ca @ la.solve(self.M(), B2)

        sys = signal.lti(A, B, C, D)

        return sys

    def freq_response(self, omega=None, modes=None):
        r"""Frequency response for a mdof system.

        This method returns the frequency response for a mdof system
        given a range of frequencies and the modes that will be used.

        Parameters
        ----------
        omega : array, optional
            Array with the desired range of frequencies (the default
             is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        omega : array
            Array with the frequencies
        magdb : array
            Magnitude (dB) of the frequency response for each pair input/output.
            The order of the array is: [output, input, magnitude]
        phase : array
            Phase of the frequency response for each pair input/output.
            The order of the array is: [output, input, phase]

        Examples
        --------
        """
        rows = self.H.inputs  # inputs (mag and phase)
        cols = self.H.inputs  # outputs

        B = self.H.B
        C = self.H.C
        D = self.H.D

        evals, psi = self._eigen(self.w, sparse=False)
        psi_inv = la.inv(psi)  # TODO change to get psi_inv from la.eig

        # if omega is not given, define a range
        # TODO adapt this range
        if omega is None:
            omega = np.linspace(0, 3000, 5000)

        # if modes are selected:
            if modes is not None:
                n = self.ndof  # n dof -> number of modes
                m = len(modes)  # -> number of desired modes
                # idx to get each evalue/evector and its conjugate
                idx = np.zeros((2 * m), int)
                idx[0:m] = modes  # modes
                idx[m:] = range(2 * n)[-m:]  # conjugates (see how evalues are ordered)

                evals_m = evals[np.ix_(idx)]
                psi_m = psi[np.ix_(range(2 * n), idx)]
                psi_inv_m = psi_inv[np.ix_(idx, range(2 * n))]

                magdb_m = np.empty((cols, rows, len(omega)))
                phase_m = np.empty((cols, rows, len(omega)))

                for wi, w in enumerate(omega):
                    diag = np.diag([1 / (1j * w - lam) for lam in evals_m])
                    H = C @ psi_m @ diag @ psi_inv_m @ B + D

                    magh = 20.0 * np.log10(abs(H))
                    angh = np.rad2deg((np.angle(H)))

                    magdb_m[:, :, wi] = magh
                    phase_m[:, :, wi] = angh

                return omega, magdb_m, phase_m

        magdb = np.empty((cols, rows, len(omega)))
        phase = np.empty((cols, rows, len(omega)))

        for wi, w in enumerate(omega):
            diag = np.diag([1 / (1j * w - lam) for lam in evals])
            H = C @ psi @ diag @ psi_inv @ B + D

            magh = 20.0 * np.log10(abs(H))
            angh = np.rad2deg((np.angle(H)))

            magdb[:, :, wi] = magh
            phase[:, :, wi] = angh

        return omega, magdb, phase

    def time_response(self, F, t, ic=None):
        r"""Time response for a rotor.

        This method returns the time response for a rotor
        given a force, time and initial conditions.

        Parameters
        ----------
        F : array
            Force array (needs to have the same length as time array).
        t : array
            Time array.
        ic : array, optional
            The initial conditions on the state vector (zero by default).

        Returns
        -------
        t : array
            Time values for the output.
        yout : array
            System response.
        xout : array
            Time evolution of the state vector.


        Examples
        --------
        """
        if ic is not None:
            return signal.lsim(self.H, F, t, ic)
        else:
            return signal.lsim(self.H, F, t)

    def plot_rotor(self, ax=None):
        """ Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples:

        """
        plt.rcParams['figure.figsize'] = (10, 5)
        plt.rcParams['xtick.labelsize'] = 0
        plt.rcParams['ytick.labelsize'] = 0

        #  define a color palette for the self
        r_pal = {'shaft': '#525252',
                 'node': '#6caed6',
                 'disk': '#bc625b',
                 'bearing': '#355d7a',
                 'seal': '#77ACA2'}

        if ax is None:
            ax = plt.gca()

        #  plot shaft centerline
        shaft_end = self.nodes_pos[-1]
        ax.plot([-.2 * shaft_end, 1.2 * shaft_end], [0, 0], 'k-.')
        try:
            max_diameter = max([disk.o_d for disk in self.disk_elements])
        except ValueError:
            max_diameter = max([shaft.o_d for shaft in self.shaft_elements])

        ax.set_ylim(-1.2 * max_diameter, 1.2 * max_diameter)
        ax.axis('equal')

        #  plot nodes
        for node, position in enumerate(self.nodes_pos):
            ax.plot(position, 0,
                    zorder=2, ls='', marker='D', color=r_pal['node'], markersize=10, alpha=0.6)
            ax.text(position, 0,
                    '%.0f' % node,
                    size='smaller',
                    horizontalalignment='center',
                    verticalalignment='center')

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            position = self.nodes_pos[sh_elm.n]
            sh_elm.patch(ax, position)

        # plot disk elements
        for disk in self.disk_elements:
            zpos = self.nodes_pos[disk.n]
            ypos = disk.i_d
            D = disk.o_d
            hw = disk.width / 2  # half width

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
            ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=r_pal['disk']))
            ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=r_pal['disk']))

        # plot bearings
        for bearing in self.bearing_seal_elements:
            # name is used here because classes are not import to this module
            if type(bearing).__name__ == 'BearingElement':
                zpos = self.nodes_pos[bearing.n]
                #  TODO this will need to be modified for tapppered elements
                #  check if the bearing is in the last node
                ypos = -self.nodes_o_d[bearing.n]
                h = -0.75 * ypos  # height

                #  node (x pos), outer diam. (y pos)
                bearing_points = [[zpos, ypos],  # upper
                                  [zpos + h / 2, ypos - h],
                                  [zpos - h / 2, ypos - h],
                                  [zpos, ypos]]
                ax.add_patch(mpatches.Polygon(bearing_points, color=r_pal['bearing']))

            elif type(bearing).__name__ == 'SealElement':
                zpos = self.nodes_pos[bearing.n]
                #  check if the bearing is in the last node
                ypos = self.nodes_o_d[bearing.n]
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
                ax.add_patch(mpatches.Polygon(seal_points_u, facecolor=r_pal['seal']))
                ax.add_patch(mpatches.Polygon(seal_points_l, facecolor=r_pal['seal']))

        # restore rc parameters after plotting
        mpl.rcParams.update(_orig_rc_params)

        return ax

    def campbell(self, speed_rad, freqs=6, mult=[1], plot=True, ax=None):
        r"""Calculates the Campbell diagram.

        This function will calculate the damped natural frequencies
        for a speed range.

        Parameters
        ----------
        speed_rad: array
            Array with the speed range in rad/s.
        freqs: int, optional
            Number of frequencies that will be calculated.
            Default is 6.
        mult: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        plot: bool, optional
            If the campbell will be plotted.
            If plot=False, points for the Campbell will be returned.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        points: array
            Array with the natural frequencies corresponding to each speed
             of the speed_rad array. It will be returned if plot=False.
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples
        --------
        >>> rotor1 = rotor_example()
        >>> speed = np.linspace(0, 400, 101)
        >>> camp = campbell(rotor1, speed, plot=False)
        >>> np.round(camp[:, 0], 1) #  damped natural frequencies at the first rotor speed (0 rad/s)
        array([  82.7,   86.7,  254.5,  274.3,  679.5,  716.8])
        >>> np.round(camp[:, 10], 1) # damped natural frequencies at 40 rad/s
        array([  82.6,   86.7,  254.3,  274.5,  676.5,  719.7])
        """
        rotor_state_speed = self.w

        speed_rad = np.array(speed_rad)
        z = []  # will contain values for each whirl (0, 0.5, 1)
        points_all = np.zeros([freqs, len(speed_rad)])

        for idx, w0, w1 in(zip(range(len(speed_rad)),
                               speed_rad[:-1],
                               speed_rad[1:])):
            # define shaft speed
            # check rotor state to avoid recalculating eigenvalues
            if not self.w == w0:
                self.w = w0

            # define x as the current speed and y as each wd
            x_w0 = np.full_like(range(freqs), w0)
            y_wd0 = self.wd[:freqs]

            # generate points for the first speed
            points0 = np.array([x_w0, y_wd0]).T.reshape(-1, 1, 2)
            points_all[:, idx] += y_wd0  # TODO verificar teste

            # go to the next speed
            self.w = w1
            x_w1 = np.full_like(range(freqs), w1)
            y_wd1 = self.wd[:freqs]
            points1 = np.array([x_w1, y_wd1]).T.reshape(-1, 1, 2)

            new_segment = np.concatenate([points0, points1], axis=1)

            if w0 == speed_rad[0]:
                segments = new_segment
            else:
                segments = np.concatenate([segments, new_segment])

            whirl_w = [whirl(self.kappa_mode(wd)) for wd in range(freqs)]
            z.append([whirl_to_cmap(i) for i in whirl_w])

        if plot is False:
            return points_all

        z = np.array(z).flatten()

        cmap = ListedColormap([c_pal['blue'],
                               c_pal['grey'],
                               c_pal['red']])

        if ax is None:
            ax = plt.gca()
        lines_2 = LineCollection(segments, array=z, cmap=cmap)
        ax.add_collection(lines_2)

        # plot harmonics in hertz
        for m in mult:
            ax.plot(speed_rad, m * speed_rad,
                    color=c_pal['green2'],
                    linestyle='dashed')

        # axis limits
        ax.set_xlim(0, max(speed_rad))
        ax.set_ylim(0, max(max(mult) * speed_rad))

        # legend and title
        ax.set_xlabel(r'Rotor speed ($rad/s$)')
        ax.set_ylabel(r'Damped natural frequencies ($rad/s$)')

        forward_label = mpl.lines.Line2D([], [],
                                         color=c_pal['blue'],
                                         label='Forward')
        backwardlabel = mpl.lines.Line2D([], [],
                                         color=c_pal['red'],
                                         label='Backward')
        mixedlabel = mpl.lines.Line2D([], [],
                                      color=c_pal['grey'],
                                      label='Mixed')

        ax.legend(handles=[forward_label, backwardlabel, mixedlabel],
                  loc=2)

        # restore rotor speed
        self.w = rotor_state_speed

        return ax

    def plot_time_response(self, F, t, dof, ax=None):
        """Plot the time response.

        This function will take a rotor object and plot its time response
        given a force and a time.

        Parameters
        ----------
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        dof : int
            Degree of freedom that will be observed.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples:
        ---------
        """
        t_, yout, xout = self.time_response(F, t)

        if ax is None:
            ax = plt.gca()

        ax.plot(t, yout[:, dof])

        if dof % 4 == 0:
            obs_dof = '$x$'
            amp = 'm'
        elif dof + 1 % 4 == 0:
            obs_dof = '$y$'
            amp = 'm'
        elif dof + 2 % 4 == 0:
            obs_dof = '$\alpha$'
            amp = 'rad'
        else:
            obs_dof = '$\beta$'
            amp = 'rad'

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (%s)' % amp)
        ax.set_title('Response for node %s and degree of freedom %s'
                     % (dof//4, obs_dof))

        return ax

    # TODO add frequency response - see vtoolbox

    def save_mat(self, file_name):
        """
        Save matrices and rotor model to a .mat file.
        """
        dic = {'M': self.M(),
               'K': self.K(),
               'C': self.C(),
               'G': self.G(),
               'nodes': self.nodes_pos}

        sio.savemat('%s/%s.mat' % (os.getcwd(), file_name), dic)


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
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [ShaftElement(l, i_d, o_d, steel,
                               shear_effects=True,
                               rotary_inertia=True,
                               gyroscopic=True) for l in L]

    disk0 = DiskElement(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def MAC(u, v):
    """MAC for two vectors"""
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v)**2 / ((H(u) @ u)*(H(v) @ v)))


def MAC_modes(U, V, n=None, plot=True):
    """MAC for multiple vectors"""
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])

    if not plot:
        return macs

    xpos, ypos = np.meshgrid(range(n), range(n))
    xpos, ypos = 0.5 + xpos.flatten(), 0.5 + ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(xpos)
    dy = 0.75 * np.ones_like(xpos)
    dz = macs.T.flatten()

    fig = plt.figure(figsize=(12, 8))
    #fig.suptitle('MAC - %s vs %s' % (U.name, V.name), fontsize=12)
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=plt.cm.viridis(dz), alpha=0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    #ax.set_xlabel('%s  modes' % U.name)
    #ax.set_ylabel('%s  modes' % V.name)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=1))
    # fake up the array of the scalar mappable
    sm._A = []
    cbar = fig.colorbar(sm, shrink=0.5, aspect=10)
    cbar.set_label('MAC')

    return macs


def whirl(kappa_mode):
    """Evaluates the whirl of a mode"""
    if all(kappa >= -1e-3 for kappa in kappa_mode):
        whirldir = 'Forward'
    elif all(kappa <= 1e-3 for kappa in kappa_mode):
        whirldir = 'Backward'
    else:
        whirldir = 'Mixed'
    return whirldir


def whirl_to_cmap(whirl):
    """Maps the whirl to a value"""
    if whirl == 'Forward':
        return 0
    elif whirl == 'Backward':
        return 1
    else:
        return 0.5


