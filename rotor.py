import numpy as np
import scipy.linalg as la

class Rotor(object):
    """A rotor object.

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

    Returns
    ----------
    A rotor object.

    Examples:

    """

    def __init__(self, shaft_elements, disk_elements, bearing_elements):
        self.shaft_elements = shaft_elements
        self.bearing_elements = bearing_elements
        self.disk_elements = disk_elements
        #  number of dofs
        self.ndof = 4 * len(shaft_elements) + 4

        #  ========== Assembly the rotor matrices ==========

        #  Create the matrices
        M0 = np.zeros((self.ndof, self.ndof))
        C0 = np.zeros((self.ndof, self.ndof))
        G0 = np.zeros((self.ndof, self.ndof))
        K0 = np.zeros((self.ndof, self.ndof))

        #  Skew-symmetric speed dependent contribution to element stiffness matrix
        #  from the internal damping.
        #  TODO add the contribution for K1 matrix
        K1 = np.zeros((self.ndof, self.ndof))

        #  Shaft elements
        for elm in shaft_elements:
            node = elm.n
            n1 = 4 * node   # first dof
            n2 = n1 + 8     # last dof

            M0[n1:n2, n1:n2] += elm.M()
            G0[n1:n2, n1:n2] += elm.G()
            K0[n1:n2, n1:n2] += elm.K()

        #  TODO Add error for elements with the same n (node)

        #  Disk elements
        for elm in disk_elements:
            node = elm.n
            n1 = 4 * node
            n2 = n1 + 4     # Disk is inserted in the dofs (4) of the first node

            M0[n1:n2, n1:n2] += elm.M()
            G0[n1:n2, n1:n2] += elm.G()

        for elm in bearing_elements:
            node = elm.n
            n1 = 4 * node
            n2 = n1 + 2 # Simple bearing

            C0[n1:n2, n1:n2] += elm.C()
            K0[n1:n2, n1:n2] += elm.K()
            #  TODO implement this for bearing with mode dofs

        # creates the state space matrix
        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)
        Minv = la.pinv(M0)

        A = np.vstack([np.hstack([Z, I]),
                       np.hstack([-Minv @ K0, -Minv @ C0])])

        self.A = A
        self.M = M0
        self.C = C0
        self.G = G0
        self.K = K0

    @staticmethod
    def index(eigenvalues):
        """
        Function used to generate an index that will sort
        eigenvalues and eigenvectors.
        """
        # positive in increasing order
        idxp = eigenvalues.imag.argsort()[int(len(eigenvalues)/2):]
        # negative in decreasing order
        idxn = eigenvalues.imag.argsort()[int(len(eigenvalues)/2) - 1::-1]

        idx = np.hstack([idxp, idxn])

        #  TODO implement sort that considers the cross of eigenvalues
        return idx

    def eigen(self):
        """
        This method will return the eigenvalues and eigenvectors of the
        state space matrix A.
        """
        return la.eig(self.A)

    def eigen_sorted(self):
        """
        This method will return the eigenvalues and eigenvectors of the
        state space matrix A sorted by the index method.
        """
        evalues, evectors = la.eig(self.A)
        idx = self.index(evalues)
        return evalues[idx], evectors[:, idx]
